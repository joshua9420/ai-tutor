# Tutor/Reviewer App

A local AI-powered tutor and reviewer application that:
1. **Uploads** PDFs and extracts their text.
2. **Embeds** text chunks using [Ollama](https://github.com/jmorganca/ollama) (e.g. `mxbai-embed-large` model).
3. **Stores** the chunks in [Qdrant](https://qdrant.tech/) for vector search.
4. Provides a **Gradio** web interface to:
   - Display an **outline** of the material (optionally generated by an LLM).
   - **Study** by reading concise summaries of selected text sections.
   - **Test** by generating multiple-choice questions (or other quiz items).
   - **Navigate** chunks (with “Previous” / “Next”) in a Chunk Viewer.

---

## Features

- **PDF Ingestion**  
  Extract text from PDFs using [pdfplumber] or a similar library.
  
- **Text Chunking**  
  Split large texts into smaller, overlapping chunks (configurable size/overlap).

- **Local Embeddings**  
  Use [Ollama Python API](https://github.com/ollama/ollama-python) to generate embeddings with a **local** model (e.g., `mxbai-embed-large`).

- **Vector Storage in Qdrant**  
  Upsert chunk embeddings + payload (`text`) into Qdrant, enabling semantic search.

- **Outline Generation**  
  (Optional) Generate an overall outline for the extracted text if it fits in your LLM’s context window.

- **Study & Test Modes**  
  - **Study**: Summaries of the selected text chunk.
  - **Test**: Multiple-choice or short-answer questions, generated by an LLM.

- **Chunk Navigation**  
  “Previous” and “Next” buttons allow you to iterate through stored chunks in Qdrant.

- **Gradio UI**  
  Easy-to-use web app with file upload, text boxes, and button triggers.

---

## Requirements

- **Python** ≥ 3.8
- **Ollama** Python package and CLI  
  - macOS (official) or Linux (experimental).  
  - `pip install ollama`  
  - [Ollama CLI installation](https://ollama.ai/docs/installation)  
  - Pull your model: `ollama pull mxbai/embed-large`
- **Qdrant** (≥ 1.12 recommended)  
  - E.g. via Docker:  
    ```bash
    docker run -p 6333:6333 qdrant/qdrant
    ```
  - Confirm the Qdrant dashboard at [http://localhost:6333/dashboard](http://localhost:6333/dashboard).
- **Python Libraries**  
  - `gradio`
  - `langchain` (if you still use certain splitting utilities)
  - `pdfplumber` (or `PyPDF2`) for PDF ingestion
  - `qdrant-client`
  - `ollama`
  - See [`requirements.txt`](#example-requirementstxt)

---

## Example `requirements.txt`

```txt
gradio==3.31.0
langchain==0.0.200
qdrant-client==1.12.5
pdfplumber==0.9.0
ollama==0.0.11
```


Reasoned for a couple of seconds
markdown
Copy
Edit
# Tutor/Reviewer App

A local AI-powered tutor and reviewer application that:
1. **Uploads** PDFs and extracts their text.
2. **Embeds** text chunks using [Ollama](https://github.com/jmorganca/ollama) (e.g. `mxbai-embed-large` model).
3. **Stores** the chunks in [Qdrant](https://qdrant.tech/) for vector search.
4. Provides a **Gradio** web interface to:
   - Display an **outline** of the material (optionally generated by an LLM).
   - **Study** by reading concise summaries of selected text sections.
   - **Test** by generating multiple-choice questions (or other quiz items).
   - **Navigate** chunks (with “Previous” / “Next”) in a Chunk Viewer.

---

## Features

- **PDF Ingestion**  
  Extract text from PDFs using [pdfplumber] or a similar library.
  
- **Text Chunking**  
  Split large texts into smaller, overlapping chunks (configurable size/overlap).

- **Local Embeddings**  
  Use [Ollama Python API](https://github.com/ollama/ollama-python) to generate embeddings with a **local** model (e.g., `mxbai/embed-large`).

- **Vector Storage in Qdrant**  
  Upsert chunk embeddings + payload (`text`) into Qdrant, enabling semantic search.

- **Outline Generation**  
  (Optional) Generate an overall outline for the extracted text if it fits in your LLM’s context window.

- **Study & Test Modes**  
  - **Study**: Summaries of the selected text chunk.
  - **Test**: Multiple-choice or short-answer questions, generated by an LLM.

- **Chunk Navigation**  
  “Previous” and “Next” buttons allow you to iterate through stored chunks in Qdrant.

- **Gradio UI**  
  Easy-to-use web app with file upload, text boxes, and button triggers.

---

## Requirements

- **Python** ≥ 3.8
- **Ollama** Python package and CLI  
  - macOS (official) or Linux (experimental).  
  - `pip install ollama`  
  - [Ollama CLI installation](https://ollama.ai/docs/installation)  
  - Pull your model: `ollama pull mxbai/embed-large`
- **Qdrant** (≥ 1.12 recommended)  
  - E.g. via Docker:  
    ```bash
    docker run -p 6333:6333 qdrant/qdrant
    ```
  - Confirm the Qdrant dashboard at [http://localhost:6333/dashboard](http://localhost:6333/dashboard).
- **Python Libraries**  
  - `gradio`
  - `langchain` (if you still use certain splitting utilities)
  - `pdfplumber` (or `PyPDF2`) for PDF ingestion
  - `qdrant-client`
  - `ollama`
  - See [`requirements.txt`](#example-requirementstxt)

---

## Example `requirements.txt`

```txt
gradio==3.31.0
langchain==0.0.200
qdrant-client==1.12.5
pdfplumber==0.9.0
ollama==0.0.11
(Versions are examples; use latest stable releases if preferred.)
```

## Installation & Setup
### Clone or Download this repo:

```bash
git clone https://github.com/joshua9420/ai-tutor.git
cd ai-tutor
```

### Create a Virtual Environment (recommended):
```bash
python -m venv venv
source venv/bin/activate    # macOS/Linux
# or .\venv\Scripts\activate (Windows)
```

### Install Python Dependencies:

```bash
pip install -r requirements.txt
```

### Install & Run Qdrant

Docker method:
```bash
docker run -p 6333:6333 qdrant/qdrant
```
Ensure Qdrant’s dashboard is accessible at http://localhost:6333/dashboard.

### Install & Setup Ollama

Follow Ollama docs to install on macOS or Linux.
Then pull your embedding model:
```bash
ollama pull mxbai-embed-large
```

## Usage
### Launch the App

```bash
python app.py
```
or if your main script is named differently, adjust accordingly.

### Open the Gradio UI

By default, it might show a local URL (e.g. http://127.0.0.1:7860).
Visit it in your browser.

### Upload a PDF

Click “Upload your PDF” and select a file.
The app will:
Extract text from the PDF,
Generate embeddings,
Store chunks in Qdrant.

### Outline Generation (optional)

The system might generate an outline for the entire text if configured.
The outline appears in the “Generated Outline” textbox.

### Chunk Viewer

Use “Previous Chunk” and “Next Chunk” buttons to cycle through embedded chunks.
Each chunk’s text is shown in the “Chunk Viewer.”

### Study / Test

Enter or paste text/chunk you want to study into the “Paste a section from the outline here...” field.
Click “Study” → summarization.
Click “Test” → question generation.
Architecture Overview

### Ingestion

We parse the PDF (via pdfplumber) and produce raw text.
Optional: generate an outline with an LLM.

### Chunking

Use RecursiveCharacterTextSplitter or a heading-based approach to segment the text into smaller pieces.

### Embeddings

Each chunk is passed to the Ollama Python API (ollama.embeddings(...)) with model mxbai/embed-large.
We get a list of floats representing the embedding.

### Vector DB (Qdrant)

We upsert (vector, payload={"text": chunk}, id=...) to Qdrant.
Qdrant stores vectors for semantic search.

### Retrieval

If you do a search or query, the app obtains an embedding for the query and performs a similarity search in Qdrant.

### Study & Test

Summaries and question generation calls a local or API-based LLM, passing the chunk’s text plus user instructions.

### UI (Gradio)

The front-end shows file upload, a text box for the outline, chunk navigation, and study/test interactions.

## Frequently Asked Questions
Q: Why does Ollama say “No module named ‘ollama’”?
A: Ensure your environment is correct (macOS/Linux) and that pip install ollama was done in the same Python interpreter.

Q: Why do I see “Connection Refused” to Qdrant?
A: Make sure Docker is running, the container is started with -p 6333:6333, and you’re using the correct URL (http://localhost:6333).

Q: How big can my PDFs be?
A: Large PDFs will be chunked. Generating an entire outline in one shot might exceed context limits. If so, do multi-pass chunk-based outlines.