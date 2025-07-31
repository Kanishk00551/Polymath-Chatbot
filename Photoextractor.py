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

# Import for pysqlite3
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


# --- CONFIGURATION ---

# Remove SSL cert errors if present
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Constants for directories and models
CHROMA_DIR = "chroma_memory_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
PDF_DIR = os.path.abspath("uploaded_pdfs")
PDF_IMAGE_DIR = os.path.abspath("pdf_images")

# Create directories if they don't exist
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(PDF_IMAGE_DIR, exist_ok=True)


# --- INITIALIZATION ---

# Initialize LLM, embedder, and database
llm = ChatGroq(model="gemma2-9b-it", groq_api_key=GROQ_API_KEY)
embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedder)


# --- HELPER FUNCTIONS ---

def analyze_image_with_llava(image_bytes: bytes, prompt: str) -> str | None:
    """
    Analyzes an image using the LLaVA model and returns the description.
    """
    img_base64 = base64.b64encode(image_bytes).decode()
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llava", "prompt": prompt, "images": [img_base64]},
            stream=True,
            timeout=60
        )
        if response.status_code == 200:
            vision_output = ""
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode("utf-8"))
                    vision_output += chunk.get("response", "")
            return vision_output.strip()
        else:
            st.error(f"LLaVA API request failed with status code {response.status_code}: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the LLaVA server. Please ensure it is running and accessible.")
        return None
    except requests.exceptions.Timeout:
        st.error("The request to the LLaVA server timed out. The server might be overloaded or running slow.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during image analysis: {e}")
        return None


def extract_images_from_pdf(pdf_path: str, output_dir: str) -> list[str]:
    """
    Extracts images from a PDF file.
    """
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


# --- STREAMLIT UI ---

st.set_page_config(page_title="Chroma Chatbot", page_icon="🧠")
st.title("🧠 Polymath Chatbot")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "file_status" not in st.session_state:
    st.session_state.file_status = {}

# --- SIDEBAR FOR CONTROLS ---
with st.sidebar:
    st.header("Controls")
    if st.button("Clear Memory and Start New Chat"):
        st.session_state.messages = []
        st.session_state.file_status = {}
        if os.path.exists(CHROMA_DIR):
            db.delete_collection()
            db.persist()
        st.rerun()

# --- IMAGE ANALYSIS FROM UPLOADED IMAGE ---
st.header("Upload and Analyze Image")
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    prompt = "Describe this image in detail."

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()

    with st.spinner("Analyzing image with LLaVA..."):
        vision_output = analyze_image_with_llava(image_bytes, prompt)
        if vision_output:
            st.success("Image description from LLaVA:")
            st.markdown(f"**{vision_output}**")
            vision_doc = Document(page_content=vision_output, metadata={"role": "vision"})
            db.add_documents([vision_doc])
            db.persist()
            st.session_state.messages.append({"role": "assistant", "content": f"Image description: {vision_output}"})

# --- PDF PROCESSING ---
st.header("Upload PDF Documents")
uploaded_pdfs = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if uploaded_pdfs:
    for uploaded_file in uploaded_pdfs:
        if st.session_state.file_status.get(uploaded_file.name) != "processed":
            st.info(f"Processing new file: {uploaded_file.name}")
            file_path = os.path.join(PDF_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Process text from PDF
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
                st.error(f"Error processing text from {uploaded_file.name}: {e}")

            # Process images from PDF
            image_output_dir = os.path.join(PDF_IMAGE_DIR, os.path.splitext(uploaded_file.name)[0])
            extracted_images = extract_images_from_pdf(file_path, image_output_dir)

            if extracted_images:
                st.info(f"Analyzing {len(extracted_images)} image(s) from {uploaded_file.name}...")
                for idx, img_path in enumerate(extracted_images):
                    with st.spinner(f"Processing image {idx+1}/{len(extracted_images)}..."):
                        with open(img_path, "rb") as img_file:
                            image_bytes = img_file.read()
                        vision_output = analyze_image_with_llava(image_bytes, "Describe this image from a PDF.")
                        if vision_output:
                            vision_doc = Document(
                                page_content=f"PDF Image Description: {vision_output}",
                                metadata={"role": "vision", "source": uploaded_file.name}
                            )
                            db.add_documents([vision_doc])
                            db.persist()

            st.session_state.file_status[uploaded_file.name] = "processed"
            st.success(f"Finished processing {uploaded_file.name}")

# --- CHATBOT SECTION ---
st.header("Chat with PDF & Image Memory")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask anything about the uploaded PDFs or images")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        relevant_docs = db.similarity_search(user_input, k=5)
        text_context = "\n---\n".join([doc.page_content for doc in relevant_docs if doc.metadata.get("role") != "vision"])
        image_context = "\n---\n".join([doc.page_content for doc in relevant_docs if doc.metadata.get("role") == "vision"])

        final_prompt = f"""
You are a helpful AI assistant. Your task is to answer the user's question based on the provided context.

**Instructions:**
1.  Review the "Text Context" and "Relevant Image Descriptions" provided below.
2.  Synthesize the information from both sources to formulate a comprehensive answer.
3.  If the context contains the answer, use it as the primary basis for your response.
4.  If the context does not contain the answer, explicitly state that you cannot answer based on the provided documents. **Do not use your general knowledge.**

**Text Context:**
{text_context}

**Relevant Image Descriptions:**
{image_context}

**User Question:**
{user_input}

**Answer:**
"""
        with st.spinner("Thinking..."):
            response = llm.invoke([HumanMessage(content=final_prompt)])
            reply = response.content

            with st.chat_message("assistant"):
                st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

    except Exception as e:
        st.error("An error occurred during the chat process.")
        st.exception(e)
    except Exception as e:
        st.error("Unexpected failure during chat handling.")
        st.exception(e)
