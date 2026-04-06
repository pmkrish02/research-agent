# Deep Research Agent

A research agent built in Go that compiles raw text into structured wiki articles, retrieves relevant context via embedding similarity, and answers complex queries under a 2K token budget.

## How to Run

1. Set your Gemini API key:
```bash
export GOOGLE_API_KEY="your-api-key"
```

2. Create the data directory:
```bash
mkdir -p data/raw
```

3. Run the server:
```bash
go mod tidy
go run main.go
```

Server starts on `:8080`.

## API Endpoints

### `GET /health`
```bash
curl http://localhost:8080/health
```

### `POST /ingest` — Add raw text
```bash
curl -X POST http://localhost:8080/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "Photosynthesis is the process by which green plants convert sunlight, water, and carbon dioxide into glucose and oxygen. It occurs primarily in the chloroplasts of plant cells."}'
```

### `POST /compile` — Compile raw text into wiki articles
```bash
curl -X POST http://localhost:8080/compile
```

### `POST /research` — Ask a research question
```bash
curl -X POST http://localhost:8080/research \
  -H "Content-Type: application/json" \
  -d '{"question": "How do plants produce energy and what role does sunlight play?"}'
```

Response includes the answer, sub-questions, sources used, and token budget status:
```json
{
  "question": "How do plants produce energy and what role does sunlight play?",
  "sub_questions": ["How do plants produce energy?", "What role does sunlight play in plant energy production?"],
  "answer": "...",
  "sources_used": ["abc123.txt"],
  "tokens_used": 450,
  "token_budget": 2000,
  "within_budget": true
}
```

## Architecture

```
Ingest: Raw text → saved to data/raw/ with UUID filename

Compile: data/raw/* → LLM structures each file → saved to data/wiki/

Research:
    Question
        → Embed question (Gemini)
        → Embed each wiki article
        → Cosine similarity ranking
        → Select top articles within 2K token budget
        → Decompose question into sub-questions
        → LLM generates answer from context + sub-questions
        → Return answer with sources and token usage
```

## Tech Stack

- **Go** — HTTP server
- **Gemini API** — text generation (gemini-3.1-flash-lite-preview) and embeddings (gemini-embedding-001)
- **Cosine similarity** — in-memory vector comparison for retrieval
- **Karpathy-inspired wiki compilation** — raw text pre-processed into structured knowledge

## Design Decisions

See [evaluation.md](evaluation.md) for detailed architecture trade-offs, memory strategy, and future improvements.