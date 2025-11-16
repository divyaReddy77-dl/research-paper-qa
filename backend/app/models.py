from sqlalchemy import Boolean, Column, Integer, String, Text, DateTime, ForeignKey, Float
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base


class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    full_name = Column(String(200), nullable=True)
    hashed_password = Column(String(500), nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    documents = relationship("Document", back_populates="owner", cascade="all, delete-orphan")
    questions = relationship("QuestionHistory", back_populates="user", cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(500), index=True, nullable=False)
    file_path = Column(String(1000), nullable=False)
    content = Column(Text, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    file_size = Column(Integer, nullable=True)
    page_count = Column(Integer, nullable=True)
    
    owner = relationship("User", back_populates="documents")
    questions = relationship("QuestionHistory", back_populates="document", cascade="all, delete-orphan")


class QuestionHistory(Base):
    __tablename__ = "question_history"
    
    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    context = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    document = relationship("Document", back_populates="questions")
    user = relationship("User", back_populates="questions")