<img width="1899" height="663" alt="image" src="https://github.com/user-attachments/assets/5469858b-1f92-48ad-8524-d00d6ed1e9f9" />

<img width="1909" height="830" alt="image" src="https://github.com/user-attachments/assets/51e593ee-a2c9-4925-bc53-65918521ddb8" />

🧠 Polymath Chatbot — Gemini Pro Image & Text Interface
A lightweight Python script to interact with Google Gemini Pro using text and image inputs. This tool allows you to send natural language queries and optional images to Gemini Pro and receive intelligent, multimodal responses — perfect for building intelligent assistants or integrating into larger GenAI workflows.

🚀 Features
🔤 Text Input — Chat with Gemini Pro using plain text.

🖼️ Image Input — Attach a local image to get context-aware responses.

📂 Automatic Image Encoding — Converts local image files to base64 automatically.

✅ Error Handling — Checks for API key, file path, and response status.

🛠️ Minimal dependencies, simple script, fast testing.

🗂️ File Structure
vbnet
Copy
Edit
📁 Polymath-Chatbot/
│
├── Newgeminipro.py    # Main script to interact with Gemini Pro using image + text
⚙️ Requirements
Python 3.7+

google.generativeai

PIL (Python Imaging Library or pillow)

Install dependencies:

bash
Copy
Edit
pip install google-generativeai pillow
🔐 Setup
Get your API key from Google AI Studio

Set your key as an environment variable:

bash
Copy
Edit
export GOOGLE_API_KEY="your-api-key-here"
(Windows Command Prompt)

cmd
Copy
Edit
set GOOGLE_API_KEY=your-api-key-here
💡 Usage
Run the script:

bash
Copy
Edit
python Newgeminipro.py
You will be prompted to enter:

The path to a local image (e.g. test.jpg)

Your query (e.g. "What does this image represent?")

The Gemini Pro model will respond accordingly using both text and image.

🧪 Example
bash
Copy
Edit
Enter the path of the image: dog.jpg
Enter your query: What breed is this dog?
➡️ Response:

css
Copy
Edit
This appears to be a Labrador Retriever, known for its friendly nature and intelligence.
🧠 Powered By
Google Generative AI SDK

Gemini Pro (Vision Multimodal Model)

📌 Notes
Ensure your image path is correct and the file exists.

The script uses model.generate_content() with both image and text inputs.

Gemini Vision Pro may take a few seconds to respond depending on image size and network.

🧾 License
This project is under the MIT License. Feel free to use and modify.
