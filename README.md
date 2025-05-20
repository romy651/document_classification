# AI Assignment

Hi there 👋,

This is the take home assignment for engineer position.

## Prerequisites

- Python 3.13 or higher
- [uv](https://github.com/astral-sh/uv) package manager

## Setup

0. Install uv (if not already installed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

1. Clone this repo

2. Install dependencies using uv:

```bash
uv sync
```

3. Run:

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

4. You can now open the docs at http://0.0.0.0:8000/docs and run the endpoints

## API Endpoints

### Docs

http://0.0.0.0:8000/docs

### Document Classification

- `POST /classify` - Submit a document to be classified

## Task

The task is to complete the /classify endpoint
The endpoint should

1. Take in a PDF file as an input - Use the sample documents provided under sample directory
2. Classify the PDF as one of

- "1040"
- "W2"
- "1099
- "ID Card"
- "Handwritten note"
- "OTHER"

3. Also parse the year the document was issued
