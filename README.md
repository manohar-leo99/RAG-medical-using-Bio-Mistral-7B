# RAG-medical-using-Bio-Mistral-7B

<div align="center">

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![LangChain](https://img.shields.io/badge/🦜_LangChain-Enabled-brightgreen)](https://langchain.com/)

_An intelligent medical question-answering system powered by Retrieval-Augmented Generation (RAG)_

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Performance](#-performance)
- [Troubleshooting](#-troubleshooting)
- [Disclaimer](#-disclaimer)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 Overview

This project implements a **Medical Retrieval-Augmented Generation (RAG)** system using a fully open-source stack. It enables users to ask medical questions and receive accurate, context-aware answers backed by relevant medical documents. The system leverages state-of-the-art language models specifically fine-tuned for biomedical text understanding.

### What is RAG?

RAG (Retrieval-Augmented Generation) combines:

- **Information Retrieval**: Finding relevant documents from a knowledge base
- **Text Generation**: Using an LLM to generate contextually accurate answers based on retrieved documents

This approach minimizes hallucinations and provides verifiable, source-backed responses.

---

## ✨ Features

- 🤖 **BioMistral-7B LLM**: Specialized medical language model for accurate biomedical responses
- 📚 **PubMedBert Embeddings**: Medical domain-specific embeddings for semantic search
- 🗄️ **Qdrant Vector Database**: High-performance, self-hosted vector storage
- 🔗 **LangChain Integration**: Seamless orchestration of RAG components
- ⚡ **FastAPI Backend**: Modern, high-performance API framework
- 🎨 **Web Interface**: User-friendly chat interface for queries
- 📄 **Source Attribution**: Answers include source document references
- 🚀 **Local Deployment**: Runs entirely on your machine with no external API calls
- 🔒 **Privacy-First**: All data processing happens locally

---

## 🏗️ Architecture

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│         FastAPI Application         │
│  ┌─────────────────────────────┐   │
│  │     LangChain RAG Chain     │   │
│  └─────────────────────────────┘   │
└─────────┬───────────────────────────┘
          │
          ├─────────────────────┐
          ▼                     ▼
┌──────────────────┐   ┌──────────────────┐
│  Qdrant Vector   │   │   BioMistral-7B  │
│    Database      │   │      (GGUF)      │
└──────────────────┘   └──────────────────┘
          ▲
          │
┌──────────────────┐
│  PubMedBert      │
│   Embeddings     │
└──────────────────┘
```

---

## 🛠️ Tech Stack

| Component      | Technology                | Purpose                       |
| -------------- | ------------------------- | ----------------------------- |
| **LLM**        | BioMistral-7B (INT4 GGUF) | Medical text generation       |
| **Embeddings** | PubMedBert                | Semantic search embeddings    |
| **Vector DB**  | Qdrant                    | Document storage & retrieval  |
| **Framework**  | LangChain                 | RAG orchestration             |
| **Backend**    | FastAPI                   | REST API server               |
| **Frontend**   | HTML/CSS/JS + Bootstrap   | User interface                |
| **Inference**  | Llama.cpp                 | CPU-optimized model inference |

---

## 📋 Prerequisites

### System Requirements

- **Operating System**: Windows 11 (tested on 23H2) or Linux
- **Processor**: Intel® Core™ Ultra 7 165H or equivalent (x86-64 CPU recommended)
- **RAM**: Minimum 16GB, 32GB+ recommended
- **Storage**: ~10GB free space
- **Python**: 3.11.9 or higher

### Software Dependencies

1. **Microsoft Visual C++ Compiler** (Windows)
   - Required for `llama-cpp-python` compilation
   - [Download and Installation Guide](https://code.visualstudio.com/docs/cpp/config-msvc)

2. **Docker Desktop**
   - [Download Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install/)
   - Enable WSL 2 backend (recommended)

3. **Git with LFS**
   ```bash
   git lfs install
   ```

---

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/Medical-RAG-using-Bio-Mistral-7B.git
cd Medical-RAG-using-Bio-Mistral-7B
```

### Step 2: Create Virtual Environment

```bash
python -m venv biomistral_rag
```

**Activate the environment:**

- **Windows:**

  ```bash
  biomistral_rag\Scripts\activate
  ```

- **Linux/Mac:**
  ```bash
  source biomistral_rag/bin/activate
  ```

### Step 3: Install Dependencies

```bash
python -m pip install pip --upgrade
pip install -r requirements.txt
pip install qdrant-client --upgrade
```

### Step 4: Download Models

#### BioMistral-7B Model (INT4 Quantized)

Download the GGUF model file (~4GB):

```bash
# Option 1: Using wget (Linux/WSL)
wget https://huggingface.co/MaziyarPanahi/BioMistral-7B-GGUF/resolve/main/BioMistral-7B.Q4_K_M.gguf

# Option 2: Using curl
curl -L -o BioMistral-7B.Q4_K_M.gguf https://huggingface.co/MaziyarPanahi/BioMistral-7B-GGUF/resolve/main/BioMistral-7B.Q4_K_M.gguf
```

Or download manually from: [BioMistral-7B GGUF](https://huggingface.co/MaziyarPanahi/BioMistral-7B-GGUF/blob/main/BioMistral-7B.Q4_K_M.gguf)

Place the file in the project root directory.

#### PubMedBert Embeddings Model

```bash
git lfs install
git clone https://huggingface.co/NeuML/pubmedbert-base-embeddings
```

### Step 5: Setup Qdrant Vector Database

#### Pull Qdrant Docker Image

```bash
docker pull qdrant/qdrant
```

#### Run Qdrant Container

```bash
docker run -p 6333:6333 -v .\qdrant_db\:/qdrant/storage qdrant/qdrant
```

**For Linux/Mac:**

```bash
docker run -p 6333:6333 -v ./qdrant_db:/qdrant/storage qdrant/qdrant
```

#### Verify Qdrant Installation

Open your browser and navigate to: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

### Step 6: Prepare Your Documents

1. Place your medical PDF documents in the `data/` folder
2. Supported formats: PDF, TXT, DOCX (add to `ingest.py` glob pattern if needed)

### Step 7: Create Vector Embeddings

```bash
python ingest.py
```

This process will:

- Load documents from the `data/` folder
- Split them into chunks (700 characters with 70 character overlap)
- Generate embeddings using PubMedBert
- Store embeddings in Qdrant

**Verify:** Check the Qdrant dashboard for the new `vector_db` collection.

### Step 8: Launch the Application

```bash
uvicorn app:app --reload
```

The application will be available at: [http://localhost:8000](http://localhost:8000)

---

## 💻 Usage

### Web Interface

1. Open your browser to `http://localhost:8000`
2. Type your medical question in the text area
3. Click "Submit"
4. View the AI-generated answer along with source document references

### Example Queries

- "What are the symptoms of diabetes?"
- "Explain the mechanism of action of metformin"
- "What are the side effects of ACE inhibitors?"
- "Describe the pathophysiology of hypertension"

### API Endpoint

You can also interact with the API directly:

```bash
curl -X POST "http://localhost:8000/get_response" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "query=What are the symptoms of diabetes?"
```

---

## 📁 Project Structure

```
Medical-RAG-using-Bio-Mistral-7B/
│
├── app.py                          # FastAPI application
├── ingest.py                       # Document ingestion and embedding creation
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── data/                           # Medical documents (PDFs, TXT)
│   └── [your medical documents]
│
├── templates/
│   └── index.html                  # Web interface
│
├── qdrant_db/                      # Qdrant vector database storage (auto-created)
│
├── pubmedbert-base-embeddings/     # Embedding model (downloaded)
│
└── BioMistral-7B.Q4_K_M.gguf      # LLM model file (downloaded)
```

---

## ⚙️ Configuration

### Model Parameters

Edit `app.py` to adjust LLM parameters:

```python
llm = LlamaCpp(
    model_path="BioMistral-7B.Q4_K_M.gguf",
    temperature=0.3,      # Lower = more focused, Higher = more creative
    max_tokens=2048,      # Maximum response length
    top_p=1               # Nucleus sampling parameter
)
```

### Retrieval Settings

Adjust the number of retrieved documents:

```python
retriever = db.as_retriever(search_kwargs={"k": 1})  # Change k for more/fewer sources
```

### Text Chunking

Modify chunk size in `ingest.py`:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,      # Characters per chunk
    chunk_overlap=70     # Overlap between chunks
)
```

---

## 📊 Performance

### Reference Implementation

**Test Environment:**

- **CPU**: Intel® Core™ Ultra 7 165H
- **RAM**: 64GB
- **OS**: Windows 11 Pro 23H2
- **Python**: 3.11.9

**Benchmarks:**

- Average response time: 15-30 seconds (depending on query complexity)
- Embedding generation: ~5-10 seconds per document
- Memory usage: ~8-12GB RAM during inference

### Optimization Tips

1. **Use INT4 quantization** (already included) for faster inference
2. **Adjust `n_gpu_layers`** in LlamaCpp if you have a GPU
3. **Increase CPU threads** with `n_threads` parameter
4. **Reduce `max_tokens`** for faster but shorter responses

---

## 🔧 Troubleshooting

### Common Issues

#### Issue: `llama-cpp-python` installation fails

**Solution:**

- Ensure Microsoft Visual C++ Build Tools are installed
- Try: `pip install llama-cpp-python --no-cache-dir`

#### Issue: Docker container won't start

**Solution:**

- Check if port 6333 is already in use: `netstat -ano | findstr :6333`
- Kill conflicting process or use different port

#### Issue: Out of memory errors

**Solution:**

- Reduce `max_tokens` in app.py
- Close other applications
- Use a smaller model variant (Q2 or Q3 quantization)

#### Issue: Slow response times

**Solution:**

- Ensure Docker has adequate resources (Settings → Resources)
- Use SSD instead of HDD for vector database
- Reduce retrieval depth (`k` parameter)

#### Issue: "Model file not found"

**Solution:**

- Verify `BioMistral-7B.Q4_K_M.gguf` is in the correct directory
- Update `model_path` in `app.py` if using different location

---

## ⚠️ Disclaimer

> **IMPORTANT**: This application is a **technology demonstration** for exploring AI/ML capabilities in the medical domain. It is **NOT** intended for:
>
> - Clinical decision-making
> - Medical diagnosis
> - Treatment recommendations
> - Professional medical advice
>
> **Always consult qualified healthcare professionals for medical guidance.**
>
> The developers assume no liability for any consequences resulting from the use of this software.

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Areas for Contribution

- 📄 Additional medical document formats support
- 🎨 UI/UX improvements
- 🧪 Unit tests and integration tests
- 📊 Performance optimizations
- 🌐 Multi-language support
- 📱 Mobile-responsive design

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **BioMistral Team** - For the medical LLM
- **HuggingFace** - Model hosting and transformers library
- **Qdrant** - High-performance vector database
- **LangChain** - RAG orchestration framework
- **Intel** - Reference implementation on Intel AI PC

---

## 📞 Support

If you encounter issues or have questions:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Open an [Issue](https://github.com/yourusername/Medical-RAG-using-Bio-Mistral-7B/issues)
3. Review existing issues for solutions

---

<div align="center">

**Built with ❤️ using Open Source Technologies**

[⬆ Back to Top](#-medical-rag-using-biomistral-7b)

</div>
   ![image](https://github.com/user-attachments/assets/5ec90875-be78-4bc9-9acf-09859235e313)    
#### Sample Outputs
![image](https://github.com/user-attachments/assets/94282267-ebf0-46eb-b587-996e886e6cb7)   
![image](https://github.com/user-attachments/assets/37aa70b3-1b6d-49b3-a050-d6d5aef882ee)
