import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import base64
import requests
from PIL import Image
import io
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

import os
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]

# Load API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Constants
CHROMA_DIR = "chroma_memory_db"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Initialize LLM (Groq + Gemma2)
llm = ChatGroq(model="gemma2-9b-it", groq_api_key=GROQ_API_KEY)

# Streamlit UI setup
st.set_page_config(page_title="Chroma Chatbot", page_icon="🧠")
st.title("🧠 Polymath Chatbot ")

# Embedding model
embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# Load or create Chroma vector store
db = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedder
)

# Initialize chat history in session state if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Image Recognition Section
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Define prompt here
    prompt = "Describe this image in detail."

    # Convert image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    # Call LLaVA API
    with st.spinner("Analyzing image with LLaVA..."):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llava",
                    "prompt": prompt,
                    "images": [img_base64]
                },
                stream=True  # stream is important!
            )

            if response.status_code == 200:
                vision_output = ""
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line.decode("utf-8"))
                        vision_output += chunk.get("response", "")

                st.success("Image description from LLaVA:")
                st.markdown(f"**{vision_output.strip()}**")

                # Save to Chroma memory
                vision_doc = Document(page_content=vision_output, metadata={"role": "vision"})
                db.add_documents([vision_doc])
                db.persist()

                # Add LLaVA output to chat history as an AI message
                st.session_state.messages.append({"role": "assistant", "content": f"Image description from LLaVA: {vision_output.strip()}"})

            else:
                st.error(f"LLaVA failed with status {response.status_code}: {response.text}")

        except Exception as e:
            st.error(f"LLaVA call crashed: {str(e)}")

# PDF Processing Section
st.header("PDF Document Upload")
# In photobotgemini.py

PDF_DIR = os.path.abspath("uploaded_pdfs") # Changed to absolute path
os.makedirs(PDF_DIR, exist_ok=True)



uploaded_pdfs = st.file_uploader("Upload PDF Dcouments ",type = "PDF",accept_multiple_files=True)
if uploaded_pdfs:
    st.info(f"Processing {len(uploaded_pdfs)}PDF(s)")
    for uploaded_file in uploaded_pdfs:
        file_path = os.path.join(PDF_DIR,uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Saved {uploaded_file.name}")


    # Process all PDFs in the directory
    with st.spinner("Creating embeddings for PDFs..."):
        try:
            loader = PyPDFDirectoryLoader(PDF_DIR)
            docs = loader.load()
            if docs:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                # You might want to process all documents in the directory, not just[:50]
                final_documents = text_splitter.split_documents(docs)

                #Add pdf to same Chroma db instance
                db.add_documents(final_documents)
                db.persist()
                st.success(f"Successfully embedded {len(final_documents)} chunks from PDFs into ChromaDB.")
                st.session_state.messages.append({"role": "assistant", "content": f"Successfully loaded and embedded {len(final_documents)} PDF chunks into memory."})
            else:
                st.warning("No new PDF documents found or loaded from directory.")
        except Exception as e:
            st.error(f"Error processing PDFs: {e}")
            st.exception(e)

# Chat inference
st.header("Chat with Memory")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Ask me anything about the images or PDFs")

if user_input:
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Store user query in Chroma
    user_doc = Document(page_content=user_input, metadata={"role": "user"})
    db.add_documents([user_doc])
    db.persist()

    # Retrieve top 3 relevant past chats
    relevant_docs = db.similarity_search(user_input, k=3)

    # Format chat history for context
    chat_history = []
    for doc in relevant_docs:
        content = doc.page_content
        role = doc.metadata.get("role", "user")
        if role == "user":
            chat_history.append(HumanMessage(content=content))
        else:
            chat_history.append(AIMessage(content=content))

    # Add current user input for LLM context
    chat_history.append(HumanMessage(content=user_input))

    # Generate response
    with st.spinner("Thinking..."):
        try:
            response = llm.invoke(chat_history)
            reply = response.content

            # Display response
            with st.chat_message("assistant"):
                st.markdown(reply)

            # Store response in Chroma
            reply_doc = Document(page_content=reply, metadata={"role": "assistant"})
            db.add_documents([reply_doc])
            db.persist()

            # Add AI response to chat history
            st.session_state.messages.append({"role": "assistant", "content": reply})

        except Exception as e:
            st.error(f"Error calling Groq API: {e}") # Changed error message
            st.exception(e) # This will print the full traceback in the Streamlit UI for debugging