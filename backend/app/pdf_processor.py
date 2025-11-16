import fitz  # PyMuPDF
from pdfminer.high_level import extract_text as pdfminer_extract
import pdfplumber
from typing import Optional


class PDFProcessor:
    def _init_(self):
        pass
    
    def extract_text_pymupdf(self, pdf_path: str) -> str:
        """Extract text using PyMuPDF (fitz)."""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            print(f"PyMuPDF extraction failed: {e}")
            return None
    
    def extract_text_pdfminer(self, pdf_path: str) -> str:
        """Extract text using pdfminer.six."""
        try:
            text = pdfminer_extract(pdf_path)
            return text
        except Exception as e:
            print(f"PDFMiner extraction failed: {e}")
            return None
    
    def extract_text_pdfplumber(self, pdf_path: str) -> str:
        """Extract text using pdfplumber."""
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            print(f"PDFPlumber extraction failed: {e}")
            return None
    
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from PDF using multiple methods.
        Tries PyMuPDF first, falls back to others if needed.
        """
        # Try PyMuPDF first (fastest)
        text = self.extract_text_pymupdf(pdf_path)
        
        if not text or len(text.strip()) < 100:
            print("PyMuPDF extraction insufficient, trying pdfplumber...")
            text = self.extract_text_pdfplumber(pdf_path)
        
        if not text or len(text.strip()) < 100:
            print("pdfplumber extraction insufficient, trying pdfminer...")
            text = self.extract_text_pdfminer(pdf_path)
        
        if not text or len(text.strip()) < 100:
            raise ValueError("Unable to extract sufficient text from PDF")
        
        return self.clean_text(text)
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove page numbers (simple heuristic)
        lines = text.split('\n')
        cleaned_lines = [line for line in lines if not line.strip().isdigit()]
        
        return '\n'.join(cleaned_lines)