# 📚 Research Paper QA System

AI-powered Question Answering system for research papers using NLP and transformers.

## 🎯 Features
- PDF upload and text extraction
- AI-powered question answering
- Semantic search using embeddings
- Answer highlighting with context
- Question history tracking
- User authentication
- Admin dashboard

## 🛠️ Tech Stack
- **Backend**: Python, FastAPI, SQLAlchemy
- **Frontend**: React.js, Axios
- **NLP**: Transformers (RoBERTa), Sentence-BERT
- **Database**: SQLite
- **Authentication**: JWT

## 🚀 Installation

### Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
python -m app.utils_create_admin
python -m uvicorn app.main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm start
```

## 📝 Usage
1. Register/Login
2. Upload PDF research paper
3. Ask questions about the content
4. Get AI-generated answers with context

## 👤 Default Admin
- Username: `admin`
- Password: `admin123`

## 📄 License
MIT