# Evaluation: Deep Research Agent Architecture

## Memory Architecture: Wiki Compilation + Embedding Retrieval

This agent uses a two-phase memory architecture inspired by [Andrej Karpathy's LLM OS concept](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f), where raw information is pre-processed into structured knowledge before query time.

**Phase 1 — Compile (write-time):** Raw text is ingested and compiled into structured wiki articles using an LLM. This is a summarization cascade — the LLM organizes, extracts key concepts, definitions, and relationships once. The cost is paid upfront, not on every query.

**Phase 2 — Research (query-time):** When a question arrives, the agent embeds the question, computes cosine similarity against wiki article embeddings, and retrieves only the most relevant articles within a token budget. The question is decomposed into sub-questions, and the LLM generates an answer grounded in the retrieved context.

### Why Wiki Compilation Over Raw RAG

A traditional RAG pipeline chunks raw text, embeds chunks, and retrieves them at query time. This works well for large corpora but has a key weakness: raw text is noisy. Irrelevant details, repetitive passages, and unstructured formatting waste context tokens.

Wiki compilation solves this by having the LLM pre-process raw text into dense, structured articles. The result is that each token of context sent to the LLM at query time carries more information density than raw text would. This is especially valuable under tight token constraints.

The trade-off: compilation is lossy. The LLM may discard details during compilation that turn out to be relevant later. Raw RAG preserves everything. For this use case — operating under a 2K token budget — the density gain outweighs the information loss.

## Constraint: 2,000 Token Budget

The agent enforces a self-imposed limit of approximately 2,000 tokens of context per query. This is enforced **before** the LLM call, not checked after the fact.

### How Enforcement Works

1. Articles are ranked by cosine similarity to the query.
2. The agent iterates through ranked articles, estimating tokens as `len(content) / 4`.
3. Articles are added to the context until the budget would be exceeded.
4. If no article fits within budget, the top article is truncated to fit — ensuring at least one source is always included.

### Why 2,000 Tokens

This constraint demonstrates that the agent can operate under meaningful resource limits. In production, this could map to cost constraints (e.g., $0.05 per session) or latency requirements where shorter prompts mean faster responses.

If a question genuinely requires more context, the architecture could be extended with chunk-level retrieval within wiki articles rather than whole-article retrieval. This would allow finer-grained context selection within the same budget.

## Retrieval: In-Memory Cosine Similarity

The agent computes cosine similarity between the query embedding and wiki article embeddings using Gemini's `gemini-embedding-001` model. This was chosen over a vector database (e.g., Weaviate) for several reasons:

- **No external dependency** — the agent is self-contained, no database setup required
- **Lower latency for small corpora** — in-memory comparison avoids network round-trips
- **Simplicity** — for a research agent with tens to low hundreds of articles, a vector database adds complexity without proportional benefit

The trade-off: this approach does not scale beyond a few hundred articles. At that point, a vector database with approximate nearest neighbor search would be necessary.

## Question Decomposition

Complex research questions are broken into 2-3 focused sub-questions by the LLM before the main answer is generated. This serves two purposes:

1. **Better retrieval** — sub-questions can surface different aspects of the relevant context
2. **Structured reasoning** — the LLM addresses each facet of the question explicitly rather than attempting a monolithic answer

## Design Philosophy: Why Not Standard RAG?

Most take-home assignments for research agents default to a standard RAG pipeline: chunk → embed → store in vector DB → retrieve → generate. This works, but it treats all text as equal.

The Karpathy-inspired wiki compilation approach makes a deliberate architectural choice: **spend more at write-time to save at read-time.** The LLM acts as an editor during ingestion, distilling raw text into structured knowledge with clear concepts, definitions, and relationships. This means every token retrieved at query time carries maximum information density.

This mirrors how human researchers actually work — you don't re-read raw papers every time you need an answer. You build notes, summaries, and mental models first, then query your structured understanding. The compile step is the machine equivalent of that process.

The trade-off is explicit: compilation is lossy, and the system depends on the LLM's judgment about what's important. But under a tight token budget, dense structured context consistently outperforms raw text retrieval.

## Business Impact & Cost/Benefit Analysis

### The Problem This Solves

Research teams and knowledge workers spend hours reading, synthesizing, and cross-referencing documents to answer complex questions. An internal knowledge base with hundreds of documents becomes unusable without intelligent retrieval — people either read everything (slow) or search by keyword (misses context).

This agent turns a pile of unstructured text into a queryable knowledge base with a single API call. A team could ingest internal docs, customer call transcripts, or competitive research and get grounded, cited answers in seconds.

### Cost Analysis

The agent makes 3 types of LLM API calls per research query:

| Call | Model | Estimated Cost |
|------|-------|---------------|
| Question embedding | gemini-embedding-001 | ~$0.001 |
| Article embeddings (N articles) | gemini-embedding-001 | ~$0.001 × N |
| Question decomposition | gemini-3.1-flash-lite-preview | ~$0.002 |
| Answer generation (2K context) | gemini-3.1-flash-lite-preview | ~$0.003 |

For a knowledge base of 20 wiki articles, a single research query costs roughly **$0.03**. The compile step is a one-time cost of ~$0.01-0.02 per raw document.

### Why the Token Budget Matters for Business

The 2K token constraint isn't just a technical exercise — it directly maps to cost control. In a production SaaS product, unbounded context means unbounded cost. A customer asking 1,000 questions against a large knowledge base could generate significant API bills. The token budget creates a predictable, per-query cost ceiling that enables sustainable pricing.

The wiki compilation step further reduces ongoing costs. By pre-processing raw text into dense structured articles, every query retrieves fewer tokens for the same information value. Over thousands of queries, this compounds into meaningful savings compared to raw RAG where chunks carry noise and redundancy.

### Where This Creates Value

- **Customer support teams** — ingest product docs + past tickets, answer customer questions with cited sources
- **Research analysts** — ingest reports, earnings calls, news articles, query across all of them
- **Internal knowledge management** — company wikis that actually answer questions instead of requiring manual search
- **Due diligence** — ingest contracts or regulatory documents, ask targeted questions under cost constraints

## Known Limitations and Future Improvements

**Current limitations:**

- **Embeddings are computed at query time for articles.** Every research query embeds all wiki articles, which is expensive and slow. In production, embeddings should be pre-computed during the `/compile` phase and cached to disk. Query time would then only require one embedding call (for the question).

- **No chunking within articles.** If a single wiki article exceeds the token budget, it is truncated rather than intelligently chunked. Implementing chunk-level retrieval within articles would improve precision.

- **No concurrent compilation.** Wiki articles are compiled sequentially. Go's goroutines and channels could parallelize this with a semaphore pattern to respect API rate limits.

- **Text-only ingestion.** The system only handles plain text. Supporting PDFs, markdown, and other formats would make it practical for real research workflows.

- **No function calling / tool use.** The current flow is hardcoded: embed → search → decompose → answer. With function calling, the LLM could dynamically decide which tools to use — `search_wiki`, `read_article`, `summarize`, `answer` — adapting its strategy to the question. A simple question might skip decomposition entirely; a complex one might iterate through multiple search-read-summarize cycles. This would transform the system from a script into a true agent.

**What I would add with more time:**

1. Pre-computed embedding cache saved during compilation
2. Chunk-level retrieval within wiki articles
3. Concurrent compilation with goroutine-based parallelism
4. Function calling so the LLM chooses its own research strategy
5. Session-level cost tracking to enforce per-session budget constraints
6. Multi-format document support (PDF, markdown)ß