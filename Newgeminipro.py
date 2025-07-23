import os
import streamlit as st
import base64
import requests
import io
import json
import fitz  # PyMuPDF
from PIL import Image
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage

# Remove SSL cert errors if present
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]

# Load API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Constants
CHROMA_DIR = "chroma_memory_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
PDF_DIR = os.path.abspath("uploaded_pdfs")
PDF_IMAGE_DIR = os.path.abspath("pdf_images")

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(PDF_IMAGE_DIR, exist_ok=True)

# Initialize components
llm = ChatGroq(model="gemma2-9b-it", groq_api_key=GROQ_API_KEY)
embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedder)

# Streamlit UI
st.set_page_config(page_title="Chroma Chatbot", page_icon="🧠")
st.title("🧠 Polymath Chatbot")

# NEW: Initialize session state to track processed files
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- IMAGE ANALYSIS FROM UPLOADED IMAGE ---
st.header("Upload and Analyze Image")
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # This section remains unchanged
    # (Image analysis logic for single image upload)
    pass # Placeholder for brevity, your original code goes here

# --- PDF PROCESSING ---
st.header("Upload PDF Documents")
uploaded_pdfs = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

def extract_images_from_pdf(pdf_path, output_dir):
    doc = fitz.open(pdf_path)
    os.makedirs(output_dir, exist_ok=True)
    saved_images = []
    for page_num in range(len(doc)):
        images = doc.get_page_images(page_num)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_p{page_num + 1}_{img_index}.{image_ext}"
            image_path = os.path.join(output_dir, image_filename)
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            saved_images.append(image_path)
    return saved_images

if uploaded_pdfs:
    for uploaded_file in uploaded_pdfs:
        # NEW: Check if this file has already been processed
        if uploaded_file.name not in st.session_state.processed_files:
            st.info(f"Starting analysis of new file: {uploaded_file.name}")
            
            # Indent all your existing processing logic under this if-statement
            file_path = os.path.join(PDF_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Saved {uploaded_file.name}")

            # Text processing
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                if docs:
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = splitter.split_documents(docs)
                    for chunk in chunks:
                        chunk.metadata["source"] = uploaded_file.name
                    db.add_documents(chunks)
                    db.persist()
                    st.success(f"Embedded {len(chunks)} text chunks from {uploaded_file.name}.")
            except Exception as e:
                st.error(f"Error embedding text from {uploaded_file.name}.")
                st.exception(e)

            # Image processing
            image_output_dir = os.path.join(PDF_IMAGE_DIR, os.path.splitext(uploaded_file.name)[0])
            extracted_images = extract_images_from_pdf(file_path, image_output_dir)

            if extracted_images:
                st.info(f"Analyzing {len(extracted_images)} image(s) from {uploaded_file.name}...")

            for idx, img_path in enumerate(extracted_images):
                with st.spinner(f"Processing image {idx+1}/{len(extracted_images)}..."):
                    try:
                        with open(img_path, "rb") as img_file:
                            img_base64 = base64.b64encode(img_file.read()).decode()

                        response = requests.post(
                            "http://localhost:11434/api/generate",
                            json={"model": "llava", "prompt": "Describe this image from a PDF.", "images": [img_base64]},
                            stream=True,
                            timeout=60
                        )
                        if response.status_code == 200:
                            vision_output = ""
                            for line in response.iter_lines():
                                if line:
                                    chunk = json.loads(line.decode("utf-8"))
                                    vision_output += chunk.get("response", "")
                            vision_doc = Document(page_content=f"PDF Image Description: {vision_output}", metadata={"role": "vision", "source": uploaded_file.name})
                            db.add_documents([vision_doc])
                            db.persist()
                    except Exception as e:
                        st.warning(f"Skipping image {idx+1} due to error: {e}")

            # NEW: Add the file to the set of processed files to mark it as "done"
            st.session_state.processed_files.add(uploaded_file.name)
            st.success(f"Finished analyzing and storing {uploaded_file.name}")

# --- CHATBOT SECTION ---
st.header("Chat with PDF & Image Memory")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask anything about the uploaded PDFs or images")
if user_input:
    # Your existing chat logic goes here, it remains unchanged
    pass # Placeholder for brevity, your original chat logic goes here