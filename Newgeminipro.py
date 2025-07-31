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


# This setup is critical for environments where sqlite3 needs a specific implementation
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("Successfully replaced sqlite3 with pysqlite3-binary.")
except ImportError:
    print("pysqlite3-binary not found, using standard sqlite3.")

# Remove SSL cert errors if present (useful in some corporate environments)
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]

# Load environment variables from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- CONSTANTS ---
CHROMA_DIR = "chroma_memory_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
PDF_DIR = os.path.abspath("uploaded_pdfs")
PDF_IMAGE_DIR = os.path.abspath("pdf_images")

# Create directories if they don't exist to prevent errors
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(PDF_IMAGE_DIR, exist_ok=True)

# --- GLOBAL COMPONENT INITIALIZATION ---
# Initialize LLM, embedder, and vector database once to be efficient
llm = ChatGroq(model="gemma2-9b-it", groq_api_key=GROQ_API_KEY)
embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedder)

# --- HELPER FUNCTIONS ---

def analyze_image_with_llava(image_bytes: bytes, prompt: str) -> str | None:
    """Analyzes an image with a local LLaVA model and returns the description."""
    img_base64 = base64.b64encode(image_bytes).decode()
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llava", "prompt": prompt, "images": [img_base64]},
            stream=True,
            timeout=60
        )
        response.raise_for_status()  # Will raise an HTTPError for bad responses (4xx or 5xx)
        vision_output = "".join(
            json.loads(line.decode("utf-8")).get("response", "")
            for line in response.iter_lines() if line
        )
        return vision_output.strip()
    except requests.exceptions.ConnectionError:
        st.error("Connection to LLaVA server failed. Is it running at http://localhost:11434?")
        return None
    except requests.exceptions.Timeout:
        st.error("LLaVA server request timed out. The server may be busy or slow.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred with the LLaVA request: {e}")
        return None

def extract_images_from_pdf(pdf_path: str, output_dir: str) -> list[str]:
    """Extracts images from a PDF and saves them to a directory."""
    saved_images = []
    try:
        doc = fitz.open(pdf_path)
        os.makedirs(output_dir, exist_ok=True)
        for page_num in range(len(doc)):
            for img_index, img in enumerate(doc.get_page_images(page_num)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_p{page_num+1}_{img_index}.{image_ext}"
                image_path = os.path.join(output_dir, image_filename)
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                saved_images.append(image_path)
    except Exception as e:
        st.error(f"Error extracting images from PDF: {e}")
    return saved_images

# --- STREAMLIT UI LAYOUT ---

st.set_page_config(page_title="Polymath Chatbot", page_icon="🧠", layout="wide")
st.title("🧠 Polymath Chatbot")

# Initialize session state variables if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = []
if "file_status" not in st.session_state:
    st.session_state.file_status = {}

# --- SIDEBAR ---
with st.sidebar:
    st.header("Controls")
    if st.button("Clear Memory & Reset Chat"):
        st.session_state.messages = []
        st.session_state.file_status = {}
        # Safely delete the collection if the directory exists
        if os.path.exists(CHROMA_DIR):
            try:
                db.delete_collection()
                db.persist()
                st.success("Memory cleared.")
            except Exception as e:
                st.error(f"Error clearing memory: {e}")
        st.rerun()

# --- MAIN CONTENT ---
col1, col2 = st.columns(2)

with col1:
    st.header("Upload & Process Files")
    # --- IMAGE UPLOADER ---
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        if st.session_state.file_status.get(uploaded_image.name) != "processed":
            st.image(uploaded_image, caption="Uploaded Image")
            with st.spinner(f"Analyzing {uploaded_image.name}..."):
                image_bytes = uploaded_image.getvalue()
                vision_output = analyze_image_with_llava(image_bytes, "Describe this image in detail.")
                if vision_output:
                    st.success("Image analysis complete.")
                    vision_doc = Document(page_content=vision_output, metadata={"role": "vision", "source": uploaded_image.name})
                    db.add_documents([vision_doc])
                    db.persist()
                    st.session_state.messages.append({"role": "assistant", "content": f"Analyzed image: {vision_output}"})
                    st.session_state.file_status[uploaded_image.name] = "processed"

    # --- PDF UPLOADER ---
    uploaded_pdfs = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
    if uploaded_pdfs:
        for pdf in uploaded_pdfs:
            if st.session_state.file_status.get(pdf.name) != "processed":
                st.info(f"Processing new file: {pdf.name}")
                file_path = os.path.join(PDF_DIR, pdf.name)
                with open(file_path, "wb") as f:
                    f.write(pdf.getbuffer())

                # Process text
                with st.spinner(f"Embedding text from {pdf.name}..."):
                    try:
                        loader = PyPDFLoader(file_path)
                        docs = loader.load()
                        if docs:
                            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                            chunks = splitter.split_documents(docs)
                            for chunk in chunks:
                                chunk.metadata["source"] = pdf.name
                            db.add_documents(chunks)
                            db.persist()
                            st.success(f"Embedded {len(chunks)} text chunks.")
                    except Exception as e:
                        st.error(f"Error processing text from {pdf.name}: {e}")

                # Process images
                with st.spinner(f"Extracting and analyzing images from {pdf.name}..."):
                    image_dir = os.path.join(PDF_IMAGE_DIR, os.path.splitext(pdf.name)[0])
                    image_paths = extract_images_from_pdf(file_path, image_dir)
                    if image_paths:
                        st.info(f"Found {len(image_paths)} images to analyze.")
                        for img_path in image_paths:
                            with open(img_path, "rb") as img_file:
                                image_bytes = img_file.read()
                            vision_output = analyze_image_with_llava(image_bytes, "Describe this image from a PDF.")
                            if vision_output:
                                doc = Document(page_content=vision_output, metadata={"role": "vision", "source": pdf.name})
                                db.add_documents([doc])
                                db.persist()
                        st.success(f"Finished analyzing images from {pdf.name}.")

                st.session_state.file_status[pdf.name] = "processed"
                st.success(f"Completed processing for {pdf.name}")

with col2:
    st.header("Chat Window")
    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if user_input := st.chat_input("Ask about the uploaded content..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    relevant_docs = db.similarity_search(user_input, k=5)
                    text_context = "\n---\n".join([doc.page_content for doc in relevant_docs if doc.metadata.get("role") != "vision"])
                    image_context = "\n---\n".join([doc.page_content for doc in relevant_docs if doc.metadata.get("role") == "vision"])

                    final_prompt = f"""
You are an AI assistant. Answer the user's question based ONLY on the context provided below.
If the context does not contain the answer, state "I cannot answer this based on the provided documents."
Do not use any external knowledge.

### Text Context:
{text_context if text_context.strip() else "No relevant text found."}

### Relevant Image Descriptions:
{image_context if image_context.strip() else "No relevant images found."}

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
                    st.exception(e) # This will print the full error traceback in the Streamlit UI
