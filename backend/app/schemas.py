from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from typing import Optional


class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: Optional[str] = None


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str]
    is_active: bool
    is_admin: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class DocumentResponse(BaseModel):
    id: int
    filename: str
    uploaded_at: datetime
    file_size: Optional[int] = None
    
    class Config:
        from_attributes = True


class DocumentDetailResponse(BaseModel):
    id: int
    filename: str
    uploaded_at: datetime
    file_size: Optional[int] = None
    page_count: Optional[int] = None
    question_count: int = 0
    
    class Config:
        from_attributes = True


class QuestionRequest(BaseModel):
    document_id: int
    question: str = Field(..., min_length=3)


class AnswerResponse(BaseModel):
    answer: str
    context: Optional[str]
    confidence: Optional[float]
    history_id: int


class HistoryResponse(BaseModel):
    id: int
    question: str
    answer: str
    context: Optional[str]
    confidence: Optional[float]
    created_at: datetime
    document_id: int
    
    class Config:
        from_attributes = True


class ContactForm(BaseModel):
    name: str = Field(..., min_length=2)
    email: EmailStr
    subject: str = Field(..., min_length=3)
    message: str = Field(..., min_length=10)