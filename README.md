<p align="center">
  <img src="https://img.shields.io/badge/📋-Payer%20Policy%20MCP%20Server-F59E0B?style=for-the-badge&labelColor=1a1a2e" alt="Payer Policy MCP Server" />
</p>

<p align="center">
  <em>A Model Context Protocol (MCP) server that enables AI agents to query insurance payer policies, prior authorization criteria, and step therapy rules using a Hybrid RAG pipeline.</em>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+" /></a>
  <a href="https://modelcontextprotocol.io/"><img src="https://img.shields.io/badge/MCP-Protocol-00C7B7?style=flat-square" alt="MCP Protocol" /></a>
  <a href="https://platform.openai.com/"><img src="https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=flat-square&logo=openai&logoColor=white" alt="OpenAI" /></a>
  <a href="https://www.pinecone.io/"><img src="https://img.shields.io/badge/Pinecone-Vector%20DB-000000?style=flat-square&logo=pinecone&logoColor=white" alt="Pinecone" /></a>
  <a href="https://huggingface.co/spaces"><img src="https://img.shields.io/badge/🤗%20HF-Spaces-FFD21E?style=flat-square" alt="Hugging Face Spaces" /></a>
  <a href="https://www.docker.com/"><img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white" alt="Docker" /></a>
  <a href="https://github.com/AryanSinha2003/Payer-Policy-Server/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License" /></a>
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-how-it-works">How It Works</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-deployment">Deployment</a> •
  <a href="#-project-structure">Project Structure</a> •
  <a href="#-tech-stack">Tech Stack</a>
</p>

---

## ✨ Features

- 📡 **MCP-Native** — Exposes a `query_pdf` tool via the [Model Context Protocol](https://modelcontextprotocol.io/), allowing any MCP-compatible AI agent to retrieve payer policy information over SSE.
- 🔍 **Hybrid RAG Pipeline** — Combines dense retrieval (Pinecone) with sparse retrieval (BM25) using Reciprocal Rank Fusion for comprehensive document search.
- ⚡ **FlashRank Re-Ranking** — CPU-optimized cross-encoder re-ranking (`ms-marco-MiniLM-L-12-v2`) for high-precision result ordering without GPU overhead.
- 📄 **Multimodal PDF Processing** — Extracts text, tables, and images from complex policy documents using `unstructured[pdf]` with Tesseract OCR support.
- 🧠 **LLM-Powered Summarization** — Automatically generates multimodal summaries for chunks containing tables or images, improving retrieval quality.
- 🐳 **Docker-Ready** — Ships with a production Dockerfile for one-command deployment on Hugging Face Spaces, Render, or any container platform.

---

## 🏗 How It Works

```
┌─────────────────────────────┐
│     AI Agent / MCP Client   │
│  (sends query via SSE)      │
└──────────────┬──────────────┘
               │
               ▼
┌──────────────────────────────┐
│    Payer Policy MCP Server   │
│         (FastMCP)            │
│                              │
│   Tool: query_pdf(query)     │
└──────────────┬───────────────┘
               │
     ┌─────────┴─────────┐
     ▼                   ▼
┌──────────┐      ┌────────────┐
│   BM25   │      │  Pinecone  │
│ (Sparse) │      │  (Dense)   │
└────┬─────┘      └─────┬──────┘
     │                  │
     └────────┬─────────┘
              ▼
   ┌─────────────────────┐
   │  Reciprocal Rank    │
   │  Fusion (RRF)       │
   └──────────┬──────────┘
              ▼
   ┌─────────────────────┐
   │  FlashRank           │
   │  Re-Ranking          │
   │  (Cross-Encoder)     │
   └──────────┬──────────┘
              ▼
   ┌─────────────────────┐
   │  OpenAI GPT 5.4 mini     │
   │  Answer Generation   │
   └─────────────────────┘
```

### RAG Pipeline Stages

| Stage | Component | Details |
|:---|:---|:---|
| **1. Retrieval** | BM25 + Pinecone | Parallel sparse (keyword) and dense (semantic) search, top-5 each |
| **2. Fusion** | Reciprocal Rank Fusion | Merges both result sets with score `1/(rank + k)` weighting |
| **3. Deduplication** | By `doc_id` | Removes duplicate chunks across retrievers |
| **4. Re-Ranking** | FlashRank | Cross-encoder re-scoring, selects top-5 most relevant passages |
| **5. Generation** | OpenAI GPT 5.4 mini | Generates answer from re-ranked context (text + tables + images) |

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Details |
|:---|:---|
| **Python** | `3.10+` |
| **OpenAI API Key** | [platform.openai.com](https://platform.openai.com/) |
| **Pinecone Account** | [pinecone.io](https://www.pinecone.io/) — Free tier works |

### 1. Clone & Install

```bash
git clone https://github.com/AryanSinha2003/Payer-Policy-Server.git
cd Payer-Policy-Server

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```env
OPENAI_API_KEY="sk-your-openai-api-key"
PINECONE_API_KEY="your-pinecone-api-key"
PINECONE_INDEX_NAME="pdf-mcp-db"
```

### 3. Set Up Pinecone Index

Create a new index in the [Pinecone Console](https://app.pinecone.io/) with the following settings:

| Parameter | Value | Why |
|:---|:---|:---|
| **Name** | `pdf-mcp-db` | Must match `PINECONE_INDEX_NAME` in `.env` |
| **Dimensions** | `768` | Matches `intfloat/e5-base-v2` embeddings |
| **Metric** | `cosine` | Similarity metric for dense retrieval |

> ⚠️ **Critical:** Dimensions **must** be `768`. Using a different value will cause ingestion and retrieval failures.

### 4. Ingest Your Policy PDF

Place your policy PDF in the project directory (default: `Prior Authorization Documentation Guide.pdf`), then run:

```bash
python ingest.py
```

**What happens during ingestion:**

```
📄 PDF Partitioning (unstructured)
    ↓
🔄 Semantic Chunking (chunk_by_title)
    ↓
🧠 Multimodal Summarization (GPT-4o for chunks with tables/images)
    ↓
🔮 Vector Embedding (intfloat/e5-base-v2)
    ↓
☁️  Upload to Pinecone (namespace: single-pdf-mcp)
    ↓
💾 Save local BM25 store (local_store.pkl)
```

> 💡 **Tip:** To use a different PDF, update the `PDF_FILE_PATH` in `config.py`.

### 5. Start the Server

```bash
python mcp_server.py
```

The server starts on `http://0.0.0.0:8000` with SSE transport. MCP clients can connect at:

```
http://localhost:8000/sse
```

---

## 🐳 Deployment

### Docker

```bash
# Build
docker build -t payer-policy-server .

# Run
docker run -p 7860:7860 \
  -e PINECONE_API_KEY="your-key" \
  -e PINECONE_INDEX_NAME="pdf-mcp-db" \
  payer-policy-server
```

The Dockerfile is based on `python:3.11-slim` and includes system dependencies for PDF processing:
- **Poppler** — PDF rendering
- **Tesseract OCR** — Text extraction from images
- **libgl1 / libglib2.0** — OpenCV dependencies

### Hugging Face Spaces

The server is pre-deployed and available at:

```
https://ary-007-payer-policy-mcp-server.hf.space/sse
```

To deploy your own instance, push this repo to a Hugging Face Space with Docker SDK enabled.

---

## 📁 Project Structure

```
Payer-Policy-Server/
├── mcp_server.py              # 📡 FastMCP server — exposes the query_pdf tool
├── rag_engine.py              # 🔍 Hybrid RAG pipeline (BM25 + Pinecone + FlashRank)
├── data_processing.py         # 📄 PDF partitioning, chunking, multimodal summarization
├── ingest.py                  # ⬆️  One-time script to ingest PDF → Pinecone + local store
├── config.py                  # ⚙️  API keys, model config, file paths
├── local_store.pkl            # 💾 Pre-built BM25 + doc store (generated by ingest.py)
├── Prior Authorization
│   Documentation Guide.pdf    # 📋 Sample payer policy document
├── Dockerfile                 # 🐳 Production container config
├── requirements.txt           # 📦 Python dependencies
└── .env                       # 🔑 Environment variables (not committed)
```

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|:---|:---|:---|
| **Server** | [FastMCP](https://github.com/jlowin/fastmcp) | MCP-compliant tool server with SSE transport |
| **Dense Retrieval** | [Pinecone](https://www.pinecone.io/) | Serverless vector similarity search |
| **Sparse Retrieval** | [BM25](https://pypi.org/project/rank-bm25/) (`rank_bm25`) | Keyword-based lexical retrieval |
| **Fusion** | Reciprocal Rank Fusion | Merges sparse + dense results by rank score |
| **Re-Ranking** | [FlashRank](https://pypi.org/project/FlashRank/) (`ms-marco-MiniLM-L-12-v2`) | CPU-optimized cross-encoder re-ranking |
| **Embeddings** | [HuggingFace](https://huggingface.co/intfloat/e5-base-v2) (`intfloat/e5-base-v2`) | 768-dim dense vector embeddings |
| **PDF Parsing** | [Unstructured](https://unstructured.io/) (`unstructured[pdf]`) | Extracts text, tables, and images from PDFs |
| **LLM** | [OpenAI GPT-4o](https://platform.openai.com/) | Answer generation + multimodal summarization |
| **Container** | [Docker](https://www.docker.com/) (`python:3.11-slim`) | Production deployment |

---

## 🔌 MCP Tool Reference

The server exposes a single tool via the Model Context Protocol:

### `query_pdf`

Query the ingested policy PDF to extract information and answer questions using Hybrid Search + GPT-4o.

**Parameters:**

| Name | Type | Required | Description |
|:---|:---|:---:|:---|
| `query` | `string` | ✅ | The natural language question about payer policies |
| `openai_api_key` | `string` | ✅ | OpenAI API key for RAG generation |

**Example (via MCP client):**

```python
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

async with sse_client(url="http://localhost:8000/sse") as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        result = await session.call_tool("query_pdf", {
            "query": "What are the step therapy requirements for GLP-1 agonists?",
            "openai_api_key": "sk-..."
        })
        print(result.content[0].text)
```

---

## 🔧 Environment Variables

| Variable | Required | Default | Description |
|:---|:---:|:---|:---|
| `OPENAI_API_KEY` | ❌* | — | Used during ingestion; at query time, passed per-request via `openai_api_key` param |
| `PINECONE_API_KEY` | ✅ | — | Pinecone API key for vector store |
| `PINECONE_INDEX_NAME` | ❌ | `pdf-mcp-db` | Name of the Pinecone index |
| `PORT` | ❌ | `8000` | Server port (set to `7860` in Docker for HF Spaces) |

> \* The OpenAI key is required for `ingest.py` (multimodal summarization). At query time, it's passed dynamically by the calling agent.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  Built with ❤️ using
  <a href="https://modelcontextprotocol.io/">MCP</a> •
  <a href="https://www.pinecone.io/">Pinecone</a> •
  <a href="https://platform.openai.com/">OpenAI</a> •
  <a href="https://unstructured.io/">Unstructured</a>
</p>
