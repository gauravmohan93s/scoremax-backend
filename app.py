from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
import requests
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

app = FastAPI()

SYSTEM_INSTRUCTION = """EXTRACT EXAM DATA. RETURN JSON.

**LAYOUT AWARENESS**:
- The document may be **SINGLE-COLUMN** or **MULTI-COLUMN**.
- Read **Column-by-Column** (down-then-right) if standard academic layout.
- If questions go across page width, read top-down.

RULES:
1. **ORPHAN CONTENT**: If text/math/options appear *before* Q1, use "question_number": "CONT".
2. **METADATA**: Extract Subject, Topic, Sub Topic, Exams, Difficulty, Tags.
3. **BOXES (0-1000)**:
   - "bounding_box": The FULL region (Question + Options + Answer Key).
   - "prompt_bounding_box": The STUDENT region (Question + Options ONLY). **EXCLUDE Answer Key/Solution.**
   - "answer_bounding_box": Visual Answer Key/Tick.
   - "solution_bounding_box": Written solution.
   - "box_mode": "normalized"
   - "box_order": "yx"
4. **QUESTION/OPTIONS SPLIT**:
   - Keep only stem in "question_text".
   - Put options only in "options" array, one option per item.
5. **ANSWER FORMAT**:
   - MCQ/MSQ: return option labels/indexes only, not full option text.
   - NAT: return numeric answer.
6. **FORMATTING**: LaTeX for Math (double-escaped \\). Markdown for Tables.
"""

USER_PROMPT_IMAGE = "Extract all questions from this page image. Handle columns correctly. Keep options out of question_text. Populate all metadata fields."

RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "questions": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "question_number": {"type": "STRING"},
                    "question_text": {"type": "STRING"},
                    "type": {"type": "STRING", "enum": ["MCQ", "MSQ", "NAT", "Matrix", "Unknown"]},
                    "options": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "detected_answer_index": {"type": "INTEGER", "nullable": True},
                    "correct_answer": {"type": "STRING", "nullable": True},
                    "solution_text": {"type": "STRING", "nullable": True},
                    "bounding_box": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                    "box_mode": {"type": "STRING", "enum": ["normalized", "absolute"]},
                    "box_order": {"type": "STRING", "enum": ["yx", "xy"]},
                    "prompt_bounding_box": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                    "stem_bounding_box": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                    "options_bounding_box": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                    "answer_bounding_box": {"type": "ARRAY", "items": {"type": "INTEGER"}, "nullable": True},
                    "solution_bounding_box": {"type": "ARRAY", "items": {"type": "INTEGER"}, "nullable": True},
                    "metadata": {
                        "type": "OBJECT",
                        "properties": {
                            "subject": {"type": "STRING"},
                            "topic": {"type": "STRING"},
                            "sub_topic": {"type": "STRING"},
                            "difficulty": {"type": "STRING", "enum": ["Easy", "Medium", "Hard"]},
                            "previous_exams": {"type": "ARRAY", "items": {"type": "STRING"}},
                            "tags": {"type": "ARRAY", "items": {"type": "STRING"}},
                        },
                        "required": ["subject", "topic"],
                    },
                },
                "required": ["question_number", "question_text", "type", "bounding_box", "prompt_bounding_box", "metadata"],
            },
        }
    },
}

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class MetadataRequest(BaseModel):
    pdfBase64: Optional[str] = None
    pdfUrl: Optional[str] = None


class ExtractRequest(BaseModel):
    imageBase64: Optional[str] = None
    pdfBase64: Optional[str] = None
    pdfUrl: Optional[str] = None
    mimeType: Optional[str] = None
    pageNumber: Optional[int] = None
    quickMode: Optional[bool] = None
    maxDim: Optional[int] = None


class ExtractResponse(BaseModel):
    questions: List[Dict[str, Any]]
    usage: Dict[str, int]
    model: str
    sourceImageWidth: int
    sourceImageHeight: int


def require_api_key(x_api_key: Optional[str], authorization: Optional[str]) -> None:
    expected = (os.getenv("EXAMAI_EXTERNAL_API_KEY") or "").strip()
    if not expected:
        return
    provided = (x_api_key or "").strip()
    if not provided and authorization:
        if authorization.lower().startswith("bearer "):
            provided = authorization.split(" ", 1)[1].strip()
    if not provided or provided != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


def resolve_gemini_key(x_gemini_api_key: Optional[str]) -> str:
    if x_gemini_api_key and x_gemini_api_key.strip():
        return x_gemini_api_key.strip()
    return (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("API_KEY")
        or ""
    ).strip()


def fetch_pdf_bytes(pdf_base64: Optional[str], pdf_url: Optional[str]) -> bytes:
    if pdf_base64:
        try:
            return base64.b64decode(pdf_base64)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid pdfBase64: {exc}")
    if pdf_url:
        try:
            resp = requests.get(pdf_url, timeout=60)
            if resp.status_code != 200:
                raise HTTPException(status_code=400, detail=f"Unable to fetch pdfUrl: {resp.status_code}")
            return resp.content
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to fetch pdfUrl: {exc}")
    raise HTTPException(status_code=400, detail="Missing pdfBase64 or pdfUrl")


def pdf_page_dimensions(pdf_bytes: bytes) -> List[Dict[str, float]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: List[Dict[str, float]] = []
    for idx in range(doc.page_count):
        page = doc.load_page(idx)
        rect = page.rect
        rotation = page.rotation
        if rotation in (90, 270):
            width = float(rect.height)
            height = float(rect.width)
        else:
            width = float(rect.width)
            height = float(rect.height)
        pages.append({"page": idx + 1, "width": width, "height": height})
    return pages


def render_page_to_jpeg(pdf_bytes: bytes, page_number: int, max_dim: int) -> Dict[str, Any]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if page_number < 1 or page_number > doc.page_count:
        raise HTTPException(status_code=400, detail=f"Invalid pageNumber {page_number}")
    page = doc.load_page(page_number - 1)
    rect = page.rect
    w = float(rect.width)
    h = float(rect.height)
    scale = 1.0
    max_dim_value = max(w, h)
    if max_dim_value > float(max_dim):
        scale = float(max_dim) / max_dim_value
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    jpeg_bytes = pix.tobytes("jpeg")
    return {
        "jpeg": jpeg_bytes,
        "width": int(pix.width),
        "height": int(pix.height),
    }


def extract_json_from_text(text: str) -> Dict[str, Any]:
    raw = text.strip()
    if not raw:
        raise ValueError("Empty response text")
    candidates = [raw]
    first = raw.find("{")
    last = raw.rfind("}")
    if first >= 0 and last > first:
        candidates.append(raw[first:last + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue
    raise ValueError("Unable to parse Gemini JSON output")


def models_to_try() -> List[str]:
    primary = (os.getenv("EXAMAI_MODEL") or "gemini-2.5-flash").strip()
    fallbacks = (os.getenv("EXAMAI_MODEL_FALLBACKS") or "gemini-2.5-flash-lite,gemini-2.0-flash").split(",")
    fallbacks = [item.strip().replace("models/", "") for item in fallbacks if item.strip()]
    combined = []
    for model in [primary] + fallbacks:
        model = model.replace("models/", "").strip()
        if model and model not in combined:
            combined.append(model)
    max_models = max(1, int(os.getenv("EXAMAI_MAX_MODELS_TO_TRY") or "2"))
    return combined[:max_models]


def call_gemini(image_base64: str, page_number: Optional[int], api_key: str) -> Dict[str, Any]:
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing Gemini API key")

    for model in models_to_try():
        url = f"{GEMINI_BASE_URL}/models/{model}:generateContent?key={api_key}"
        payload = {
            "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": RESPONSE_SCHEMA,
                "temperature": 0.1,
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": f"{USER_PROMPT_IMAGE} Page {page_number}." if page_number else USER_PROMPT_IMAGE},
                        {"inlineData": {"mimeType": "image/jpeg", "data": image_base64}},
                    ],
                }
            ],
        }
        resp = requests.post(url, json=payload, timeout=90)
        if resp.status_code == 404:
            continue
        if resp.status_code in (429, 503):
            continue
        if resp.status_code >= 400:
            raise HTTPException(status_code=500, detail=f"Gemini request failed ({resp.status_code})")
        parsed = resp.json()
        parts = parsed.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = "\n".join([p.get("text", "") for p in parts if isinstance(p.get("text", None), str)]).strip()
        data = extract_json_from_text(text)
        usage = parsed.get("usageMetadata", {}) or {}
        return {
            "questions": data.get("questions", []) if isinstance(data, dict) else [],
            "usage": {
                "promptTokens": int(usage.get("promptTokenCount") or 0),
                "candidatesTokens": int(usage.get("candidatesTokenCount") or 0),
                "totalTokens": int(usage.get("totalTokenCount") or 0),
            },
            "model": model,
        }

    raise HTTPException(status_code=500, detail="Gemini request failed for all models")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True}


@app.post("/pdf-metadata")
def pdf_metadata(
    payload: MetadataRequest,
    x_api_key: Optional[str] = Header(default=None),
    authorization: Optional[str] = Header(default=None),
):
    require_api_key(x_api_key, authorization)
    pdf_bytes = fetch_pdf_bytes(payload.pdfBase64, payload.pdfUrl)
    pages = pdf_page_dimensions(pdf_bytes)
    return {"pageDimensions": pages}


@app.post("/extract-page", response_model=ExtractResponse)
def extract_page(
    payload: ExtractRequest,
    x_api_key: Optional[str] = Header(default=None),
    authorization: Optional[str] = Header(default=None),
    x_gemini_api_key: Optional[str] = Header(default=None),
):
    require_api_key(x_api_key, authorization)
    api_key = resolve_gemini_key(x_gemini_api_key)

    image_base64 = payload.imageBase64
    source_width = 0
    source_height = 0

    if not image_base64:
        pdf_bytes = fetch_pdf_bytes(payload.pdfBase64, payload.pdfUrl)
        page_number = payload.pageNumber or 1
        max_dim = int(payload.maxDim or int(os.getenv("EXAMAI_PAGE_IMAGE_MAX_DIM", "900")))
        rendered = render_page_to_jpeg(pdf_bytes, page_number, max_dim)
        image_base64 = base64.b64encode(rendered["jpeg"]).decode("ascii")
        source_width = int(rendered["width"])
        source_height = int(rendered["height"])
    else:
        page_number = payload.pageNumber or None

    result = call_gemini(image_base64, page_number, api_key)
    return ExtractResponse(
        questions=result["questions"],
        usage=result["usage"],
        model=result["model"],
        sourceImageWidth=source_width,
        sourceImageHeight=source_height,
    )
