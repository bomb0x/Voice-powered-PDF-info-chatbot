# Voice-powered-PDF-info-chatbot

This project is designed to create a PDF-based knowledge assistant that can:

1. Automatically download relevant PDFs from a website.

2. Extract and index their content for semantic search.

3. Serve a web-based interface that allows users to:

Ask questions about the PDF content.

Transcribe audio to text.

Convert text to speech.

## Install dependencies

```
pip install -r requiements.txt
```

## Download PDFs from the given URL

```
python download_pdfs.py
```

## PDF Ingestion & Indexing

```
python ingest_pdfs.py
```

## Run FastAPI server

```
uvicorn main:app --reload
```