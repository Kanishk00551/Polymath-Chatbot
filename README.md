<img width="1899" height="663" alt="image" src="https://github.com/user-attachments/assets/5469858b-1f92-48ad-8524-d00d6ed1e9f9" />

<img width="1909" height="830" alt="image" src="https://github.com/user-attachments/assets/51e593ee-a2c9-4925-bc53-65918521ddb8" />

# Polymath Chatbot 🧠

Polymath Chatbot is a sophisticated, multi-modal AI assistant built with Python and Streamlit. It can understand and process information from both text and images contained within uploaded PDF documents, as well as from direct image uploads. The chatbot is designed to provide context-aware answers by leveraging a powerful Retrieval-Augmented Generation (RAG) system, conversational memory, and the ability to fall back on general knowledge.

---

## ✨ Features

* **Multi-Modal Input**: Accepts both PDF documents and direct image uploads (`.jpg`, `.jpeg`, `.png`).
* **PDF Text Analysis**: Extracts and embeds textual content from PDFs for semantic search.
* **Image Analysis**:
    * Extracts images embedded within PDF files.
    * Analyzes both uploaded images and extracted PDF images using the powerful **Google Gemini Vision API**.
* **Hybrid Memory System**:
    * **Long-Term Memory**: Uses **ChromaDB** as a persistent vector store to remember information from all uploaded documents.
    * **Short-Term Memory**: Remembers the last few turns of the current conversation to keep track of context (like the user's name).
* **Intelligent Response Generation**:
    * Employs a sophisticated RAG pipeline to retrieve relevant document and conversational context.
    * Uses the high-speed **Groq API** (running Gemma2) for generating text responses.
    * Follows a "context-first" approach: it first tries to answer using documents and conversation history before resorting to its general knowledge.
* **Interactive UI**: A clean and user-friendly web interface built with **Streamlit**.

---

## ⚙️ How It Works

The chatbot's workflow is designed to be a robust RAG pipeline:

1.  **File Upload**: The user uploads a PDF or an image through the Streamlit interface.
2.  **Processing**:
    * For PDFs, the application extracts both the text and any embedded images.
    * All text is split into chunks.
3.  **Embedding & Storage**:
    * Text chunks are converted into vector embeddings using a Hugging Face Sentence Transformer model (`all-MiniLM-L6-v2`).
    * Images are sent to the **Google Gemini Vision API** for detailed descriptions. These descriptions are then embedded.
    * All embeddings are stored in a local **ChromaDB** vector database.
4.  **User Interaction**: The user asks a question in the chat window.
5.  **Context Retrieval**:
    * The application performs a similarity search in ChromaDB to find the most relevant document chunks and image descriptions.
    * It also retrieves the last few messages from the current conversation history.
6.  **Prompt Engineering**: A detailed prompt is constructed, providing the LLM with the retrieved document context, image descriptions, and conversational history, along with a set of instructions on how to answer.
7.  **Response Generation**: The final prompt is sent to the **Groq API**, which generates a coherent and context-aware answer.

---

## 🚀 Setup and Installation

Follow these steps to run the Polymath Chatbot on your local machine.

### 1. Clone the Repository

```bash
git clone [https://github.com/Kanishk00551/Polymath-Chatbot.git](https://github.com/Kanishk00551/Polymath-Chatbot.git)
cd Polymath-Chatbot

📄 License
This project is licensed under the MIT License.
