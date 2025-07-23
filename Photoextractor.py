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
# CHANGED: Swapped PyPDFDirectoryLoader for the more specific PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
__import__('pysqlite3')


import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

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

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- IMAGE ANALYSIS FROM UPLOADED IMAGE ---
# This section remains unchanged
st.header("Upload and Analyze Image")
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    prompt = "Describe this image in detail."

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    with st.spinner("Analyzing image with LLaVA..."):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "llava", "prompt": prompt, "images": [img_base64]},
                stream=True,
                timeout=60 # Increased timeout for potentially slow local model
            )
            if response.status_code == 200:
                vision_output = ""
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line.decode("utf-8"))
                        vision_output += chunk.get("response", "")
                st.success("Image description from LLaVA:")
                st.markdown(f"**{vision_output.strip()}**")
                vision_doc = Document(page_content=vision_output, metadata={"role": "vision"})
                db.add_documents([vision_doc])
                db.persist()
                st.session_state.messages.append({"role": "assistant", "content": f"Image description: {vision_output.strip()}"})
            else:
                st.error(f"LLaVA failed: {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"LLaVA crashed: {str(e)}")
            st.exception(e)

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
    st.info(f"Processing {len(uploaded_pdfs)} PDF(s)...")
    for uploaded_file in uploaded_pdfs:
        file_path = os.path.join(PDF_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Saved {uploaded_file.name}")

        # === CHANGED: EFFICIENT PDF TEXT PROCESSING ===
        try:
            # Load only the specific file that was just uploaded
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            if docs:
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(docs)
                
                # NEW: Add source metadata to each chunk for better tracking
                for chunk in chunks:
                    chunk.metadata["source"] = uploaded_file.name

                db.add_documents(chunks)
                db.persist()
                st.success(f"Embedded {len(chunks)} text chunks from {uploaded_file.name}.")
        except Exception as e:
            st.error(f"Error embedding text from {uploaded_file.name}.")
            st.exception(e)
        # === END OF CHANGE ===

        image_output_dir = os.path.join(PDF_IMAGE_DIR, os.path.splitext(uploaded_file.name)[0])
        extracted_images = extract_images_from_pdf(file_path, image_output_dir)

        if extracted_images:
            st.info(f"Analyzing {len(extracted_images)} image(s) from {uploaded_file.name}...")

        for idx, img_path in enumerate(extracted_images):
            st.write(f"Processing image {idx+1}...")
            try:
                with open(img_path, "rb") as img_file:
                    image = Image.open(img_file).convert("RGB")
                    buffered = io.BytesIO()
                    image.save(buffered, format="JPEG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode()

                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": "llava", "prompt": "Describe this image from a PDF.", "images": [img_base64]},
                    stream=True,
                    timeout=60 # Increased timeout
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
                st.exception(e)

# --- CHATBOT SECTION ---
st.header("Chat with PDF & Image Memory")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask anything about the uploaded PDFs or images")
if user_input:
    try:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # === CHANGED: ROBUST CONTEXT PROMPT FOR LLM ===
        try:
            # Increase k to retrieve more potentially relevant documents
            relevant_docs = db.similarity_search(user_input, k=5) 
        except Exception as e:
            st.error("Similarity search failed.")
            st.exception(e)
            relevant_docs = []

        # Separate text and image descriptions to provide clearer context
        text_context = "\n---\n".join([doc.page_content for doc in relevant_docs if doc.metadata.get("role") != "vision"])
        image_context = "\n---\n".join([doc.page_content for doc in relevant_docs if doc.metadata.get("role") == "vision"])

        # Create a detailed prompt template for the LLM
        final_prompt = f"""
Based ONLY on the following context from a document, answer the user's question. If the context does not contain the answer, say "I cannot answer this based on the provided documents."

### Text Context:
{text_context}

### Relevant Image Descriptions:
{image_context}

### User Question:
{user_input}

### Answer:
"""
        
        try:
            # Invoke the LLM with the new, structured prompt
            response = llm.invoke([HumanMessage(content=final_prompt)])
            reply = response.content
            with st.chat_message("assistant"):
                st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            
            # Note: Saving user input and assistant replies to the vector DB can 
            # sometimes pollute the search results. Consider if this is desired behavior.
            db.add_documents([
                Document(page_content=user_input, metadata={"role": "user"}),
                Document(page_content=reply, metadata={"role": "assistant"})
            ])
            db.persist()
        except Exception as e:
            st.error("Groq API failed.")
            st.exception(e)
        # === END OF CHANGE ===

    except Exception as e:
        st.error("Unexpected failure during chat handling.")
        st.exception(e)
