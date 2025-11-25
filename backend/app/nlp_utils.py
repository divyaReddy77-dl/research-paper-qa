import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from typing import List, Tuple, Dict
import warnings

warnings.filterwarnings('ignore')


class NLPEngine:
    def __init__(self):
        """Initialize NLP models with error handling"""
        print("🚀 Loading NLP models...")
        
        # Load QA Model
        try:
            self.qa_model_name = "deepset/roberta-base-squad2"
            print(f"Loading QA model: {self.qa_model_name}")
            self.qa_tokenizer = AutoTokenizer.from_pretrained(self.qa_model_name)
            self.qa_model = AutoModelForQuestionAnswering.from_pretrained(self.qa_model_name)
            self.qa_pipeline = pipeline(
                "question-answering",
                model=self.qa_model,
                tokenizer=self.qa_tokenizer,
                device=-1  # CPU
            )
            print("✅ QA model loaded successfully")
        except Exception as e:
            print(f"⚠️ QA model failed: {e}")
            self.qa_pipeline = None
        
        # Load Embedding Model
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Embedding model loaded successfully")
        except Exception as e:
            print(f"⚠️ Embedding model failed: {e}")
            self.embedding_model = None
        
        # Document storage
        self.document_chunks: Dict[int, List[str]] = {}
        self.document_embeddings: Dict[int, np.ndarray] = {}
        self.document_metadata: Dict[int, Dict] = {}
        
        print("✅ NLP Engine ready!\n")
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks"""
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        
        if len(text.split()) <= chunk_size:
            return [text]
        
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                # Keep overlap
                overlap_sentences = current_chunk[-3:] if len(current_chunk) > 3 else current_chunk
                current_chunk = overlap_sentences.copy()
                current_length = sum(len(s.split()) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks if chunks else [text]
    
    def extract_metadata(self, text: str) -> Dict:
        """Extract publication date, authors, etc."""
        metadata = {
            'publication_date': None,
            'authors': [],
            'abstract': None,
            'keywords': []
        }
        
        # Extract publication date (multiple formats)
        date_patterns = [
            r'published[:\s]+(\d{4}[-/]\d{2}[-/]\d{2})',
            r'(\d{4}[-/]\d{2}[-/]\d{2})',
            r'([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})',
            r'\b(20\d{2})\b'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata['publication_date'] = match.group(1)
                break
        
        # Extract authors (common patterns)
        author_patterns = [
            r'(?:Author|Authors)[:\s]+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)*[A-Z][a-z]+(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z]\.?\s*)*[A-Z][a-z]+)*)',
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+and\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
        ]
        
        for pattern in author_patterns:
            matches = re.findall(pattern, text)
            if matches:
                if isinstance(matches[0], tuple):
                    metadata['authors'].extend([m for m in matches[0] if m])
                else:
                    metadata['authors'].append(matches[0])
                break
        
        # Extract abstract
        abstract_match = re.search(
            r'(?:ABSTRACT|Abstract)[:\s]+(.*?)(?:INTRODUCTION|Introduction|KEYWORDS|Keywords|\n\n[A-Z])',
            text,
            re.DOTALL | re.IGNORECASE
        )
        if abstract_match:
            metadata['abstract'] = re.sub(r'\s+', ' ', abstract_match.group(1).strip())[:1000]
        
        return metadata
    
    def process_document(self, document_id: int, text: str):
        """Process and store document with metadata"""
        print(f"📄 Processing document {document_id}...")
        
        try:
            # Extract metadata first
            self.document_metadata[document_id] = self.extract_metadata(text)
            
            # Create chunks
            chunks = self.chunk_text(text)
            self.document_chunks[document_id] = chunks
            print(f"   Created {len(chunks)} chunks")
            
            # Create embeddings
            if self.embedding_model:
                embeddings = self.embedding_model.encode(
                    chunks,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                self.document_embeddings[document_id] = embeddings
                print(f"   ✅ Created embeddings: {embeddings.shape}")
            else:
                self.document_embeddings[document_id] = None
                print(f"   ⚠️ No embeddings (model unavailable)")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            # Fallback
            self.document_chunks[document_id] = [text[:5000]]
            self.document_embeddings[document_id] = None
            self.document_metadata[document_id] = {}
    
    def find_relevant_chunks(self, document_id: int, question: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find most relevant chunks using semantic similarity"""
        if document_id not in self.document_chunks:
            raise ValueError(f"Document {document_id} not processed")
        
        chunks = self.document_chunks[document_id]
        
        # If embeddings available, use semantic search
        if self.document_embeddings.get(document_id) is not None and self.embedding_model:
            try:
                question_embedding = self.embedding_model.encode(
                    [question],
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                
                similarities = cosine_similarity(
                    question_embedding,
                    self.document_embeddings[document_id]
                )[0]
                
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                return [(chunks[idx], float(similarities[idx])) for idx in top_indices]
            
            except Exception as e:
                print(f"⚠️ Semantic search failed: {e}, using keyword search")
        
        # Fallback: keyword search
        return self._keyword_search(chunks, question, top_k)
    
    def _keyword_search(self, chunks: List[str], question: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Keyword-based search fallback"""
        question_words = set(question.lower().split())
        stop_words = {'the', 'a', 'an', 'is', 'are', 'what', 'who', 'when', 'where', 'how', 'why', 'this', 'that'}
        question_words = question_words - stop_words
        
        scores = []
        for chunk in chunks:
            chunk_words = set(chunk.lower().split())
            overlap = len(question_words & chunk_words)
            score = overlap / len(question_words) if question_words else 0
            scores.append(score)
        
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(chunks[idx], scores[idx]) for idx in top_indices]
    
    def answer_question(self, document_id: int, question: str) -> Tuple[str, str, float]:
        """
        Main question answering method
        Returns: (answer, context, confidence)
        """
        try:
            print(f"\n❓ Question: {question}")
            question_lower = question.lower()
            
            # Check metadata first for specific question types
            metadata = self.document_metadata.get(document_id, {})
            
            # PUBLICATION DATE
            if any(word in question_lower for word in ['when', 'published', 'date', 'year']):
                if metadata.get('publication_date'):
                    answer = f"The paper was published on {metadata['publication_date']}"
                    context = f"Publication information: {metadata['publication_date']}"
                    print(f"✅ Metadata answer: {answer}")
                    return answer, context, 0.95
            
            # AUTHORS
            elif any(word in question_lower for word in ['author', 'who wrote', 'written by']):
                if metadata.get('authors'):
                    authors = ', '.join(metadata['authors'])
                    answer = f"The paper was written by {authors}"
                    context = f"Author information: {authors}"
                    print(f"✅ Metadata answer: {answer}")
                    return answer, context, 0.95
            
            # ABSTRACT
            elif any(word in question_lower for word in ['abstract', 'summary']):
                if metadata.get('abstract'):
                    answer = metadata['abstract']
                    context = answer
                    print(f"✅ Metadata answer: {answer[:100]}...")
                    return answer, context, 0.95
            
            # GENERAL QUESTIONS: Use QA pipeline
            print("🔍 Searching relevant chunks...")
            chunks = self.find_relevant_chunks(document_id, question, top_k=5)
            
            # Combine top chunks
            context = ' '.join([c for c, s in chunks[:3] if s > 0.1])
            if not context or len(context) < 50:
                context = chunks[0][0]
            
            # Limit context length
            if len(context) > 4000:
                context = context[:4000]
            
            print(f"📝 Context length: {len(context)} chars")
            
            # Use QA model
            if self.qa_pipeline:
                try:
                    print("🤖 Running QA model...")
                    result = self.qa_pipeline(
                        question=question,
                        context=context,
                        max_answer_len=200,
                        handle_impossible_answer=True
                    )
                    
                    answer = result['answer'].strip()
                    confidence = max(result['score'], 0.60)
                    
                    print(f"✅ Model answer: {answer} (confidence: {confidence:.2f})")
                    
                    if len(answer) > 3 and answer != '.':
                        return answer, context[:700], confidence
                
                except Exception as e:
                    print(f"⚠️ QA model failed: {e}")
            
            # Fallback: Extract relevant sentences
            print("📋 Using sentence extraction fallback...")
            sentences = [s.strip() + '.' for s in re.split(r'[.!?]+', context) if len(s.strip()) > 20]
            if sentences:
                answer = ' '.join(sentences[:2])
                print(f"✅ Extracted answer: {answer[:100]}...")
                return answer, context[:700], 0.60
            
            return "Unable to find answer in the document.", context[:500], 0.30
        
        except Exception as e:
            print(f"❌ Error in answer_question: {e}")
            return "An error occurred while processing your question.", "", 0.0