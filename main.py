from fastapi import FastAPI, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from openai import OpenAI as OpenAIClient
from io import BytesIO

OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"

app = FastAPI()

# Serve static HTML/JS/CSS
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

# Allow frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load FAISS
INDEX_DIR = "faiss_index"
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    INDEX_DIR,
    OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
    allow_dangerous_deserialization=True
)

# Pydantic model
class Query(BaseModel):
    question: str

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    with open(file.filename, "wb") as f:
        f.write(await file.read())

    with open(file.filename, "rb") as audio_file:
        transcript = OpenAIClient(api_key=OPENAI_API_KEY).audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return JSONResponse({"text": transcript.text})

@app.post("/ask")
async def ask_question(query: Query):
    # Search similar text chunks
    docs = vectorstore.similarity_search(query.question, k=5)

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
    You are given the following context from PDFs:
    {context}

    ALWAYS answer the question using ONLY the text above. 
    Do NOT add any information not present in the text. 
    If the answer is not present, say "I don't know".

    Question: {question}
    """
    )

    # Use a small LLM chain to extract exact answer
    chain = load_qa_chain(
        OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY), 
        chain_type="stuff",
        prompt=prompt_template
    )
    answer = chain.run(input_documents=docs, question=query.question)
    return {"answer": answer}


@app.post("/tts")
async def tts(query: Query):
    response = OpenAIClient(api_key=OPENAI_API_KEY).audio.speech.create(
        model="gpt-4o-mini-tts",   # or "tts-1"
        voice="alloy",             # available: alloy, verse, orca, shimmer, ... 
        input=query.question
    )
    audio_bytes = response.read()  # get raw bytes

    audio_io = BytesIO(audio_bytes)
    return StreamingResponse(audio_io, media_type="audio/mpeg", headers={
        "Content-Disposition": "inline; filename=tts.mp3"
    })
