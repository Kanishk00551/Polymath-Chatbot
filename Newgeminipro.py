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
from langchain_core.messages import HumanMessage
import google.generativeai as genai

# --- CONFIGURATION & INITIALIZATION ---

# This setup is critical for environments where sqlite3 needs a specific implementation
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("Successfully replaced sqlite3 with pysqlite3-binary.")
except ImportError:
    print("pysqlite3-binary not found, using standard sqlite3.")

# Load environment variables from .env file or Streamlit secrets
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # Using GEMINI_API_KEY as requested

# --- CONSTANTS ---
CHROMA_DIR = "chroma_memory_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
PDF_DIR = os.path.abspath("uploaded_pdfs")
PDF_IMAGE_DIR = os.path.abspath("pdf_images")

# Create directories if they don't exist
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(PDF_IMAGE_DIR, exist_ok=True)

# --- GLOBAL COMPONENT INITIALIZATION ---
llm = ChatGroq(model="gemma2-9b-it", groq_api_key=GROQ_API_KEY)
embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedder)

# Configure the Gemini API
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("Warning: GEMINI_API_KEY not found. Image analysis will be skipped.")

# --- HELPER FUNCTIONS ---

def analyze_image_with_gemini(image_bytes: bytes, prompt: str) -> str | None:
    """
    Analyzes an image using the Google Gemini Vision API.
    """
    if not genai.api_key:
        st.error("Gemini API key is not configured. Cannot analyze images.")
        return None
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        image_part = {
            "mime_type": "image/jpeg",
            "data": image_bytes
        }
        response = model.generate_content([prompt, image_part])
        return response.text
    except Exception as e:
        st.error(f"Gemini Vision API Error: {e}")
        return None

def extract_images_from_pdf(pdf_path: str, output_dir: str) -> list[str]:
    """Extracts images from a PDF and saves them as JPEG."""
    saved_images = []
    try:
        doc = fitz.open(pdf_path)
        os.makedirs(output_dir, exist_ok=True)
        for page_num in range(len(doc)):
            for img_index, img in enumerate(doc.get_page_images(page_num)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Convert all images to JPEG for consistency with the Gemini API
                img_pil = Image.open(io.BytesIO(image_bytes))
                if img_pil.mode != "RGB":
                    img_pil = img_pil.convert("RGB")
                
                with io.BytesIO() as output:
                    img_pil.save(output, format="JPEG")
                    image_bytes = output.getvalue()

                image_filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_p{page_num+1}_{img_index}.jpeg"
                image_path = os.path.join(output_dir, image_filename)
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                saved_images.append(image_path)
    except Exception as e:
        st.error(f"PDF Image Extraction Error: {e}")
    return saved_images

# --- STREAMLIT UI ---

st.set_page_config(page_title="Polymath Chatbot", page_icon="🧠", layout="wide")
st.title("🧠 Polymath Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "file_status" not in st.session_state:
    st.session_state.file_status = {}

with st.sidebar:
    st.header("Controls")
    if st.button("Clear Memory & Reset Chat"):
        st.session_state.messages, st.session_state.file_status = [], {}
        if os.path.exists(CHROMA_DIR):
            try:
                db.delete_collection(); db.persist()
                st.success("Memory cleared.")
            except Exception as e:
                st.error(f"Error clearing memory: {e}")
        st.rerun()

col1, col2 = st.columns(2)

with col1:
    st.header("Upload & Process Files")
    # Image Uploader
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_image and st.session_state.file_status.get(uploaded_image.name) != "processed":
        st.image(uploaded_image, caption="Uploaded Image")
        with st.spinner(f"Analyzing {uploaded_image.name}..."):
            vision_output = analyze_image_with_gemini(uploaded_image.getvalue(), "Describe this image in detail.")
            if vision_output:
                st.success("Image analysis complete.")
                doc = Document(page_content=vision_output, metadata={"role": "vision", "source": uploaded_image.name})
                db.add_documents([doc]); db.persist()
                st.session_state.messages.append({"role": "assistant", "content": f"Analyzed image: {vision_output}"})
                st.session_state.file_status[uploaded_image.name] = "processed"

    # PDF Uploader
    uploaded_pdfs = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
    if uploaded_pdfs:
        for pdf in uploaded_pdfs:
            if st.session_state.file_status.get(pdf.name) != "processed":
                st.info(f"Processing new file: {pdf.name}")
                file_path = os.path.join(PDF_DIR, pdf.name)
                with open(file_path, "wb") as f:
                    f.write(pdf.getbuffer())

                with st.spinner(f"Processing {pdf.name}..."):
                    try: # Text Processing
                        docs = PyPDFLoader(file_path).load()
                        if docs:
                            chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
                            for chunk in chunks: chunk.metadata["source"] = pdf.name
                            db.add_documents(chunks); db.persist()
                            st.success(f"Embedded {len(chunks)} text chunks.")
                    except Exception as e:
                        st.error(f"Text Processing Error: {e}")
                    
                    # Image Processing
                    image_paths = extract_images_from_pdf(file_path, os.path.join(PDF_IMAGE_DIR, os.path.splitext(pdf.name)[0]))
                    if image_paths:
                        st.info(f"Found {len(image_paths)} images to analyze.")
                        for img_path in image_paths:
                            with open(img_path, "rb") as img_file:
                                vision_output = analyze_image_with_gemini(img_file.read(), "Describe this image from a PDF.")
                            if vision_output:
                                doc = Document(page_content=vision_output, metadata={"role": "vision", "source": pdf.name})
                                db.add_documents([doc]); db.persist()
                        st.success(f"Finished image analysis from {pdf.name}.")
                st.session_state.file_status[pdf.name] = "processed"

# --- CHATBOT SECTION ---
with col2:
    st.header("Chat Window")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("Ask about the uploaded content..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    relevant_docs = db.similarity_search(user_input, k=3)
                    text_context = "\n---\n".join([doc.page_content for doc in relevant_docs if doc.metadata.get("role") != "vision"])
                    image_context = "\n---\n".join([doc.page_content for doc in relevant_docs if doc.metadata.get("role") == "vision"])
                    
                    chat_history = st.session_state.messages[-5:-1]
                    formatted_chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])

                    final_prompt = f"""
You are a helpful and conversational AI assistant. Your task is to answer the user's question using the context provided below.

**Instructions:**
1.  Prioritize information from the "Document Context" and "Image Descriptions" to answer questions about uploaded files.
2.  Use the "Recent Conversation History" to remember details from the current chat, like the user's name or previous questions.
3.  Synthesize all information to provide a single, coherent answer.
4.  If you cannot find the answer in any of the provided contexts, politely say that you don't know or cannot answer.

---
### Document Context:
{text_context if text_context.strip() else "No relevant documents found."}
---
### Image Descriptions:
{image_context if image_context.strip() else "No relevant images found."}
---
### Recent Conversation History:
{formatted_chat_history if formatted_chat_history.strip() else "This is the beginning of the conversation."}
---

### User Question:
{user_input}

### Answer:
"""
                    response = llm.invoke([HumanMessage(content=final_prompt)])
                    reply = response.content
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})

                except Exception as e:
                    st.error("A critical error occurred during the chat process.")
                    st.exception(e)
