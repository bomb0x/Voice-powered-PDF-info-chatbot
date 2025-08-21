import os
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

PDF_DIR = "pdfs"
INDEX_DIR = "faiss_index"
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"

# Load PDFs
texts = []
for filename in os.listdir(PDF_DIR):
    if filename.endswith(".pdf"):
        with pdfplumber.open(os.path.join(PDF_DIR, filename)) as pdf:
            full_text = ""
            for page in pdf.pages:
                full_text += page.extract_text() + "\n"
            texts.append(full_text)

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = []
for t in texts:
    docs.extend(text_splitter.split_text(t))

# Create embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.from_texts(docs, embeddings)

# Save index
vectorstore.save_local(INDEX_DIR)
print("Index created at:", INDEX_DIR)
