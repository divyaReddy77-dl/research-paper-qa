from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import List, Optional
import os
from datetime import datetime, timedelta
import shutil

from . import models, schemas, auth, database
from .database import engine, get_db
from .pdf_processor import PDFProcessor
from .nlp_utils import NLPEngine

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Research Paper QA System",
    description="AI-powered Question Answering system for research papers",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize NLP components
pdf_processor = PDFProcessor()
nlp_engine = NLPEngine()

# Upload directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# Dependency to get current user
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    return auth.verify_token(token, credentials_exception, db)


def get_current_active_user(current_user: models.User = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


# Root endpoint
@app.get("/")
def read_root():
    return {
        "message": "Research Paper QA System API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "docs": "/docs",
            "health": "/health"
        }
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }


# Authentication endpoints
@app.post("/register", response_model=schemas.UserResponse)
def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    db_user = db.query(models.User).filter(models.User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already taken")
    
    hashed_password = auth.get_password_hash(user.password)
    db_user = models.User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password,
        full_name=user.full_name
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


@app.post("/token", response_model=schemas.Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Login and get access token"""
    user = auth.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    
    access_token = auth.create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/me", response_model=schemas.UserResponse)
def read_users_me(current_user: models.User = Depends(get_current_active_user)):
    """Get current user information"""
    return current_user


@app.put("/users/me", response_model=schemas.UserResponse)
def update_user_profile(
    user_update: schemas.UserUpdate,
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update user profile"""
    if user_update.full_name is not None:
        current_user.full_name = user_update.full_name
    if user_update.email is not None:
        # Check if email is already taken
        existing = db.query(models.User).filter(
            models.User.email == user_update.email,
            models.User.id != current_user.id
        ).first()
        if existing:
            raise HTTPException(status_code=400, detail="Email already taken")
        current_user.email = user_update.email
    
    db.commit()
    db.refresh(current_user)
    return current_user


# Document upload endpoint
@app.post("/upload", response_model=schemas.DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Upload and process a PDF research paper"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Save file
    timestamp = datetime.now().timestamp()
    safe_filename = f"{current_user.id}{int(timestamp)}{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        file_size = os.path.getsize(file_path)
        
        # Extract text from PDF
        text_content = pdf_processor.extract_text(file_path)
        
        # Create document record
        document = models.Document(
            filename=file.filename,
            file_path=file_path,
            content=text_content,
            user_id=current_user.id,
            file_size=file_size
        )
        db.add(document)
        db.commit()
        db.refresh(document)
        
        # Process document for NLP (in background)
        try:
            nlp_engine.process_document(document.id, text_content)
        except Exception as e:
            print(f"Warning: NLP processing failed: {e}")
        
        return document
        
    except Exception as e:
        # Clean up file if processing fails
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


# Question answering endpoint
@app.post("/ask", response_model=schemas.AnswerResponse)
def ask_question(
    qa_request: schemas.QuestionRequest,
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Ask a question about a document"""
    # Verify document exists and belongs to user
    document = db.query(models.Document).filter(
        models.Document.id == qa_request.document_id,
        models.Document.user_id == current_user.id
    ).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        # Get answer using NLP engine
        answer, context, confidence = nlp_engine.answer_question(
            qa_request.document_id,
            qa_request.question
        )
        
        # Save to history
        history = models.QuestionHistory(
            question=qa_request.question,
            answer=answer,
            context=context,
            confidence=confidence,
            document_id=qa_request.document_id,
            user_id=current_user.id
        )
        db.add(history)
        db.commit()
        db.refresh(history)
        
        return {
            "answer": answer,
            "context": context,
            "confidence": confidence,
            "history_id": history.id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")


# Get user's documents
@app.get("/documents", response_model=List[schemas.DocumentResponse])
def get_documents(
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get all documents for current user"""
    documents = db.query(models.Document).filter(
        models.Document.user_id == current_user.id
    ).order_by(models.Document.uploaded_at.desc()).all()
    return documents


@app.get("/documents/{document_id}", response_model=schemas.DocumentDetailResponse)
def get_document(
    document_id: int,
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get specific document details"""
    document = db.query(models.Document).filter(
        models.Document.id == document_id,
        models.Document.user_id == current_user.id
    ).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Count questions for this document
    question_count = db.query(models.QuestionHistory).filter(
        models.QuestionHistory.document_id == document_id
    ).count()
    
    return {
        **document._dict_,
        "question_count": question_count
    }


# Get question history
@app.get("/history", response_model=List[schemas.HistoryResponse])
def get_history(
    document_id: Optional[int] = None,
    limit: int = 100,
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get question history"""
    query = db.query(models.QuestionHistory).filter(
        models.QuestionHistory.user_id == current_user.id
    )
    
    if document_id:
        query = query.filter(models.QuestionHistory.document_id == document_id)
    
    history = query.order_by(
        models.QuestionHistory.created_at.desc()
    ).limit(limit).all()
    
    return history


@app.delete("/history/{history_id}")
def delete_history(
    history_id: int,
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a history entry"""
    history = db.query(models.QuestionHistory).filter(
        models.QuestionHistory.id == history_id,
        models.QuestionHistory.user_id == current_user.id
    ).first()
    
    if not history:
        raise HTTPException(status_code=404, detail="History entry not found")
    
    db.delete(history)
    db.commit()
    return {"message": "History entry deleted successfully"}


# Dashboard statistics
@app.get("/dashboard/stats")
def get_dashboard_stats(
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get dashboard statistics for current user"""
    total_documents = db.query(models.Document).filter(
        models.Document.user_id == current_user.id
    ).count()
    
    total_questions = db.query(models.QuestionHistory).filter(
        models.QuestionHistory.user_id == current_user.id
    ).count()
    
    # Recent activity (last 7 days)
    week_ago = datetime.utcnow() - timedelta(days=7)
    recent_questions = db.query(models.QuestionHistory).filter(
        models.QuestionHistory.user_id == current_user.id,
        models.QuestionHistory.created_at >= week_ago
    ).count()
    
    # Most recent document
    latest_document = db.query(models.Document).filter(
        models.Document.user_id == current_user.id
    ).order_by(models.Document.uploaded_at.desc()).first()
    
    return {
        "total_documents": total_documents,
        "total_questions": total_questions,
        "recent_questions": recent_questions,
        "latest_document": latest_document.filename if latest_document else None,
        "member_since": current_user.created_at.isoformat()
    }


# Admin endpoints
@app.get("/admin/users", response_model=List[schemas.UserResponse])
def get_all_users(
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get all users (admin only)"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    users = db.query(models.User).all()
    return users


@app.get("/admin/stats")
def get_admin_stats(
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get system statistics (admin only)"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    total_users = db.query(models.User).count()
    active_users = db.query(models.User).filter(models.User.is_active == True).count()
    total_documents = db.query(models.Document).count()
    total_questions = db.query(models.QuestionHistory).count()
    
    # Recent activity
    today = datetime.utcnow().date()
    today_questions = db.query(models.QuestionHistory).filter(
        models.QuestionHistory.created_at >= today
    ).count()
    
    return {
        "total_users": total_users,
        "active_users": active_users,
        "total_documents": total_documents,
        "total_questions": total_questions,
        "questions_today": today_questions
    }


@app.delete("/documents/{document_id}")
def delete_document(
    document_id: int,
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a document"""
    document = db.query(models.Document).filter(
        models.Document.id == document_id,
        models.Document.user_id == current_user.id
    ).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete file
    if os.path.exists(document.file_path):
        os.remove(document.file_path)
    
    # Delete from NLP engine cache
    if document_id in nlp_engine.document_chunks:
        del nlp_engine.document_chunks[document_id]
    if document_id in nlp_engine.document_embeddings:
        del nlp_engine.document_embeddings[document_id]
    
    # Delete from database (cascades to history)
    db.delete(document)
    db.commit()
    
    return {"message": "Document deleted successfully"}


# Contact form endpoint
@app.post("/contact")
async def submit_contact_form(contact: schemas.ContactForm):
    """Submit contact form (can be enhanced with email sending)"""
    # In production, you would send an email or save to database
    print(f"Contact form received from {contact.name} ({contact.email})")
    print(f"Subject: {contact.subject}")
    print(f"Message: {contact.message}")
    
    return {
        "message": "Thank you for contacting us! We'll get back to you soon.",
        "success": True
    }