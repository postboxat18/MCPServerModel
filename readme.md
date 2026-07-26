# 🚀 AI Orchestration Platform using MCP Server

An intelligent AI orchestration platform built with **MCP (Model Context Protocol) Server** that dynamically routes user requests to the most appropriate AI service based on:

- 🔒 Data confidentiality
- ⚡ Performance requirements
- 🎮 GPU availability
- 📄 Document processing
- 📚 Knowledge retrieval

The system is optimized for an **NVIDIA GTX 1650 GPU**, balancing local AI inference, document retrieval, OCR processing, and cloud-based AI services to provide fast, secure, and accurate responses.

---

# ✨ Features

- 🧠 Intelligent request routing using MCP Server
- 🔒 Local LLM inference for confidential data
- ⚡ Ultra-fast public AI responses using Groq
- 📚 RAG with pretrained ColBERT retrieval
- 📄 Intelligent OCR engine selection
- 🎮 GPU-aware resource management
- 🔄 Automatic service fallback
- 🚀 Optimized for GTX 1650 (4GB VRAM)

---

# 🏗️ System Architecture

```text
                           User Query
                                │
                                ▼
                     MCP Server (Router)
                                │
      ┌─────────────────────────┼─────────────────────────┐
      │                         │                         │
      ▼                         ▼                         ▼
 Confidential Query        General AI Query        Image / PDF
      │                         │                         │
      ▼                         ▼                         ▼
 Local Ollama               Groq API              OCR Router
      │                                                 │
      │                                          GPU Available?
      │                                         ┌──────┴──────┐
      │                                         │             │
      │                                        Yes            No
      │                                         │             │
      │                               EasyOCR / PaddleOCR  Tesseract
      │                                         │
      └──────────────────────────────┬──────────┘
                                     ▼
                             Extracted Text
                                     │
                                     ▼
                           ColBERT RAG Pipeline
                                     │
                  ┌──────────────────┴──────────────────┐
                  │                                     │
          Embedding & Indexing                 Semantic Retrieval
                  │                                     │
                  └──────────────────┬──────────────────┘
                                     ▼
                              Ollama / Groq
                                     │
                                     ▼
                              Final Response
```

---

# 🧠 MCP Server Function Routing

The MCP Server intelligently assigns functions based on the user's request.

| Request Type | Assigned Function |
|--------------|------------------|
| Confidential data | Ollama |
| General AI questions | Groq |
| OCR request | OCR Router |
| Document Question Answering | ColBERT RAG |
| Internal Knowledge Search | ColBERT RAG |
| Image Text Extraction | EasyOCR / PaddleOCR / Tesseract |

---

# 🤖 Ollama (Local AI)

Used for:

- Confidential company documents
- Internal business information
- Private knowledge bases
- Offline inference
- Sensitive AI workloads

## Why Local?

Sensitive information should never leave the local environment.

Running Ollama locally ensures:

- Data privacy
- No cloud dependency
- Secure inference
- Low-latency responses

---

# ⚡ Groq API

Used for:

- Public knowledge
- General AI questions
- Coding assistance
- Summarization
- Non-confidential requests

## Benefits

- Extremely fast inference
- Reduces local GPU usage
- Handles public AI workloads efficiently
- Improves overall response time

---

# 📚 Retrieval-Augmented Generation (RAG)

The platform uses **pretrained ColBERT** for semantic document retrieval.

Unlike traditional embedding-only retrieval, ColBERT performs **late interaction retrieval**, delivering higher search accuracy while maintaining excellent retrieval speed.

---

## RAG Workflow

```text
User Question
      │
      ▼
Document Collection
      │
      ▼
Chunking
      │
      ▼
ColBERT Indexing
      │
      ▼
Semantic Search
      │
      ▼
Relevant Chunks
      │
      ▼
LLM (Ollama / Groq)
      │
      ▼
Answer
```

---

# 🚀 Why ColBERT?

The project uses a pretrained ColBERT model because it provides:

- Better semantic retrieval
- Higher search accuracy
- Context-aware ranking
- Reduced hallucinations
- Fast document search

---

# 🎮 GPU Usage for ColBERT

ColBERT uses the GPU for:

- Document indexing
- Embedding generation
- Semantic retrieval
- Similarity computation

Since ColBERT also consumes GPU memory, GPU resources are shared intelligently with other AI services.

The MCP Server manages these workloads to avoid GPU bottlenecks.

---

# 🎮 GPU Resource Management

The platform is optimized for an NVIDIA GTX 1650 (4 GB VRAM).

Since multiple services may require GPU acceleration, the MCP Server prioritizes workloads dynamically.

## GPU Consumers

| Service | GPU Usage |
|----------|-----------|
| Ollama | High |
| ColBERT Retrieval | Medium |
| EasyOCR | Medium |
| PaddleOCR | Medium |
| Groq | None (Cloud) |
| Tesseract | CPU Only |

---

## GPU Priority

Priority order:

1. Ollama
2. ColBERT Retrieval
3. EasyOCR
4. PaddleOCR
5. Tesseract (CPU)
6. Groq (Cloud)

This scheduling minimizes GPU contention while maintaining responsive performance.

---

# 📄 Intelligent OCR Pipeline

The OCR engine is selected automatically depending on GPU availability.

## High Performance Mode

When GPU resources are available:

- EasyOCR
- PaddleOCR

Advantages:

- High accuracy
- Better multilingual support
- Faster extraction
- Handles scanned documents effectively

---

## Lightweight Mode

When Ollama or ColBERT is actively using the GPU:

- Tesseract OCR

Advantages:

- CPU-based
- Lightweight
- Fast startup
- Prevents long response delays

This ensures users are not waiting unnecessarily while GPU-intensive services are running.

---

# 🔄 Dynamic AI Routing

```text
User Query
      │
      ▼
Is Confidential?
      │
 ┌────┴────┐
 │         │
Yes        No
 │         │
 ▼         ▼
Ollama   Groq
 │
 ▼
Need Documents?
 │
 ▼
ColBERT Retrieval
 │
 ▼
Generate Response
```

---

# 🔒 Privacy Strategy

| Data Type | Processing |
|-----------|------------|
| Confidential Documents | Local Ollama |
| Internal Knowledge Base | Local ColBERT RAG |
| Public Questions | Groq API |
| OCR Processing | Local |
| Vector Search | Local |

No confidential information is transmitted to external AI providers.

---

# ⚙️ Performance Strategy

The MCP Server continuously optimizes performance by:

- Routing confidential data locally
- Sending public requests to Groq
- Using ColBERT for accurate retrieval
- Monitoring GPU availability
- Selecting the most suitable OCR engine
- Preventing GPU overload
- Reducing response latency

---

# 🛠️ Technology Stack

## AI Models

- Ollama
- Groq API

## Retrieval

- ColBERT (Pretrained)
- RAG
- Vector Index

## OCR

- EasyOCR
- PaddleOCR
- Tesseract OCR

## Backend

- Python
- MCP Server

## Hardware

- NVIDIA GTX 1650 GPU

---

# 📈 Benefits

- Intelligent AI orchestration
- Secure local inference
- Accurate document retrieval
- GPU-aware scheduling
- Automatic OCR engine selection
- Lower response latency
- Better retrieval accuracy with ColBERT
- Reduced hallucinations using RAG
- Cloud and local AI integration
- Efficient GPU utilization

---

# 🎯 Future Improvements

- Dynamic GPU memory monitoring
- Multi-agent orchestration
- Hybrid retrieval (ColBERT + Dense Embeddings)
- Streaming AI responses
- Vision Language Models (VLM)
- Automatic model selection based on confidence
- Distributed inference support

---

# 📌 Conclusion

This project demonstrates an intelligent **AI orchestration platform** powered by an **MCP Server**, designed to optimize performance, security, and user experience.

By combining:

- 🔒 **Ollama** for confidential local inference
- ⚡ **Groq** for ultra-fast public AI responses
- 📚 **Pretrained ColBERT** for high-accuracy Retrieval-Augmented Generation (RAG)
- 📄 **EasyOCR**, **PaddleOCR**, and **Tesseract** for adaptive OCR processing
- 🎮 **GPU-aware scheduling** tailored for an NVIDIA GTX 1650

the platform delivers secure, scalable, and efficient AI workflows. The MCP Server intelligently routes every request based on data sensitivity, query type, and available computing resources, ensuring users receive accurate answers with minimal latency while protecting confidential information.