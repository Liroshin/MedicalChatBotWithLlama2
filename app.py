from flask import Flask, render_template, jsonify, request

from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
from src.helper import load_pdf, text_split, download_hugging_face_embeddings
import os

app = Flask(__name__)

load_dotenv()

# Load and prepare the data
print("[INFO] Loading PDF files...")
documents = load_pdf("data/")

print("[INFO] Splitting documents into chunks...")
text_chunks = text_split(documents)

print("[INFO] Downloading HuggingFace embeddings...")
embeddings = download_hugging_face_embeddings()  # Or use HuggingFaceEmbeddings()

# texts = [t.page_content for t in text_chunks]

# Create Pinecone Vector Store
index_name = "medical"

docsearch = PineconeVectorStore.from_existing_index(index_name, embeddings)

print("[SUCCESS] Pinecone Vector Store is ready.")

PROMPT = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
chain_type_kwargs = {"prompt": PROMPT}

llm = CTransformers(model="model/llama-2-13b-chat.ggmlv3.q5_1.bin",
                    model_type="llama",
                    config={'max_new_tokens': 512, 'temperature': 0.8})


qa = RetrievalQA.from_chain_type(llm=llm,
                                chain_type="stuff",
                                retriever=docsearch.as_retriever(search_kwargs={'k':2}),
                                return_source_documents=True,
                                chain_type_kwargs=chain_type_kwargs)

@app.route("/")
def index():
    return render_template('chat.html')



if __name__ == '__main__':
    app.run(debug= False)