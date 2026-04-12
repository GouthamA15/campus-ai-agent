# 🎓 Interactive Campus Info AI Agent

An AI-powered chatbot designed to help students and faculty quickly access information from official university sources like Kakatiya University and KUCET.

---

## 🚀 Project Overview

The **Interactive Campus Info AI Agent** is a web-based chatbot that answers user queries related to:

* Academic information
* Notifications & circulars
* Certificates (Bonafide, etc.)
* Admissions & exams
* General campus-related queries

The system is designed to evolve into a **Retrieval-Augmented Generation (RAG)** pipeline using real university data.

---

## 🧠 Current Stage

✅ Basic chatbot working
❌ No scraping yet
❌ No embeddings yet

Current flow:

```
User → Chat UI → FastAPI → LLM → Response
```

---

## 🛠 Tech Stack

### Backend

* Python (FastAPI)

### AI

* Groq API (LLM)

### Frontend

* HTML
* CSS
* JavaScript (fetch API)

### Future Additions

* ChromaDB (Vector Database)
* BeautifulSoup (Web Scraping)
* RAG Pipeline

---

## 📁 Project Structure

```
campus-ai-agent/
│
├── api/
│   └── main.py              # FastAPI backend
│
├── chatbot/
│   ├── index.html          # Chat UI
│   └── script.js           # Frontend logic
│
├── assets/                 # Logos and images
│
├── .env                    # API keys
├── requirements.txt        # Dependencies
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```
git clone <your-repo-url>
cd campus-ai-agent
```

---

### 2. Install Dependencies

```
pip install -r requirements.txt
```

---

### 3. Setup Environment Variables

Create a `.env` file:

```
GROQ_API_KEY=your_api_key_here
```

---

### 4. Run Backend Server

```
uvicorn api.main:app --reload
```

Server will start at:

```
http://127.0.0.1:8000
```

---

### 5. Run Frontend

* Open `chatbot/index.html` in your browser

---

## 💬 How It Works

1. User enters a question in the chat UI
2. JavaScript sends request to FastAPI backend
3. Backend sends query to LLM (Groq)
4. LLM generates response
5. Response is displayed in chat

---

## 🔄 Future Enhancements

* 🔍 Web scraping from university websites
* 📄 PDF parsing and data extraction
* 🧠 Embedding generation
* 🗂 Vector database (ChromaDB)
* ⚡ Retrieval-Augmented Generation (RAG)
* 🔁 Automated data updates (scheduler)

---

## ⚠️ Known Limitations

* Responses are not grounded in university data yet
* No real-time updates
* No source citations

---

## 🎯 Goal

To build a **scalable, intelligent campus assistant** that provides accurate and up-to-date information directly from official university sources.

---

## 📌 Notes

* This project is being developed incrementally
* Focus is on **clean architecture + scalability**
* Avoid unnecessary complexity in early stages

---



