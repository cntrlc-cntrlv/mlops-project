# 🎓 NeuroLearn — Intelligent Q&A with LLaMA 3 + LangChain

An interactive, student-friendly **Neuro web application** powered by state-of-the-art language and retrieval models. Built using **LangChain**, **Qdrant**, and **Meta’s LLaMA 3.1 8B** via Hugging Face, this app can **understand questions**, **retrieve contextual knowledge**, and **generate clear, simple answers** — just like a real tutor.

> 🔍 Perfect for educational assistance, study support, and AI-powered learning.

---

## 🚀 Features

- ✅ Simple, responsive **Flask-based frontend**
- ✅ Embeds your custom documents using **BAAI's `bge-base-en-v1.5`**
- ✅ Stores and retrieves data via **Qdrant vector database**
- ✅ Generates human-like explanations using **Meta's `LLaMA 3.1 8B`** model
- ✅ Uses **LangChain** to orchestrate search + generation
- ✅ Optimized for **student learning and ease of understanding**

---

## 🧠 How It Works

1. **User asks a question** via the web interface.
2. **LangChain Retriever** searches relevant content in the vector database (Qdrant).
3. The **retrieved knowledge** is passed into a **prompt template**.
4. The **LLaMA 3.1 model** generates a clear, simple explanation.
5. The answer is returned in a student-friendly tone.

---

## 🧰 Tech Stack

| Layer         | Technology                           |
|---------------|---------------------------------------|
| Frontend      | HTML, CSS (via Flask templates)& js       |
| Backend       | Python, Flask                         |
| Vector Store  | Qdrant                                |
| Embeddings    | `BAAI/bge-base-en-v1.5`               |
| Language Model| `meta-llama/Llama-3.1-8B` (Hugging Face) |
| Pipeline Mgmt | LangChain                             |

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

  **bash**
  git clone https://github.com/your-username/ai-tutor-app.git
  cd ai-tutor-app '''
  

### 2. Install Requirements.txt
  pip install -r requirements.txt


### 3. Replace API Tokens
  Open server.py and replace :
  Qdrant_APIkey = "YOUR_QDRANT_API_KEY"
  HF_token = "YOUR_HUGGINGFACE_TOKEN"

  🔐 Note: Get your tokens from:
    Hugging Face: https://huggingface.co/settings/tokens
    Qdrant Cloud: https://cloud.qdrant.io/


## 4. Run The App
  python app.py
  
  Open your browser and go to :
  http://127.0.0.1:5000


---


## 📌 Example Use Case
  Question:
    Who is Voltaire?
  
  AI Tutor Response:
  
    Voltaire was a French writer and philosopher known for his wit and criticism of injustice. He believed in freedom of speech and religion and helped shape the Enlightenment. Think of him 
    like an 18th-century influencer for truth and fairness!


---


## 🧪 Model Details

  🔹 Embedding Model:
  
    Name: BAAI/bge-base-en-v1.5
    Purpose: Transforms text into high-quality embeddings for semantic similarity.
    Framework: sentence-transformers

🔹 LLM:

    Model: meta-llama/Llama-3.1-8B
    Provider: Hugging Face
    Inference Type: 4-bit quantized using bitsandbytes for memory efficiency
    Pipeline: transformers.pipeline("text-generation")




