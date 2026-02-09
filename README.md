# ScoreMax Backend (Extractor)

This is the external AI extractor service for ScoreMax. It renders PDF pages with PyMuPDF and calls Gemini to extract questions.

## Endpoints

- `GET /health`
- `POST /pdf-metadata`
  - Body: `{ "pdfBase64"?: string, "pdfUrl"?: string }`
  - Returns: `{ pageDimensions: [{ page, width, height }] }`
- `POST /extract-page`
  - Body: `{ imageBase64?, pdfBase64?, pdfUrl?, pageNumber?, maxDim?, quickMode? }`
  - Returns: `{ questions, usage, model, sourceImageWidth, sourceImageHeight }`

## Environment

- `GEMINI_API_KEY` (required unless you send `x-gemini-api-key`)
- `EXAMAI_MODEL` (optional, default `gemini-2.5-flash`)
- `EXAMAI_MODEL_FALLBACKS` (optional)
- `EXAMAI_MAX_MODELS_TO_TRY` (optional)
- `EXAMAI_EXTERNAL_API_KEY` (optional, enable to protect with `x-api-key` / `Authorization: Bearer`)
- `EXAMAI_PAGE_IMAGE_MAX_DIM` (optional, default `900`)

## Render

Build command:
```
pip install -r requirements.txt
```

Start command:
```
uvicorn app:app --host 0.0.0.0 --port $PORT
```

Add environment variables in Render dashboard.
