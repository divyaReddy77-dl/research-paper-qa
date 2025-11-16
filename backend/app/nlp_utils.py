import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from typing import List, Tuple
import warnings

warnings.filterwarnings('ignore')


class NLPEngine:
    def __init__(self):
        print("Loading NLP models...")
        
        try:
            self.qa_model_name = "deepset/roberta-base-squad2"
            print(f"Loading QA model: {self.qa_model_name}")
            self.qa_tokenizer = AutoTokenizer.from_pretrained(self.qa_model_name)
            self.qa_model = AutoModelForQuestionAnswering.from_pretrained(self.qa_model_name)
            self.qa_pipeline = pipeline(
                "question-answering",
                model=self.qa_model,
                tokenizer=self.qa_tokenizer,
                device=-1
            )
            print("✓ QA model loaded successfully")
            
        except Exception as e:
            print(f"Warning: {e}")
            self.qa_pipeline = None
        
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✓ Embedding model loaded successfully")
        except Exception as e:
            print(f"Warning: {e}")
            self.embedding_model = None
        
        self.document_chunks = {}
        self.document_embeddings = {}
        print("✓ NLP Engine ready!")
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        text = re.sub(r'\s+', ' ', text).strip()
        
        if len(text.split()) <= chunk_size:
            return [text]
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                overlap_sentences = current_chunk[-5:] if len(current_chunk) > 5 else current_chunk
                current_chunk = overlap_sentences.copy()
                current_length = sum(len(s.split()) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks if chunks else [text]
    
    def process_document(self, document_id: int, text: str):
        print(f"Processing document {document_id}...")
        
        try:
            chunks = self.chunk_text(text)
            self.document_chunks[document_id] = chunks
            print(f"Created {len(chunks)} chunks")
            
            if self.embedding_model:
                embeddings = self.embedding_model.encode(chunks, show_progress_bar=False)
                self.document_embeddings[document_id] = embeddings
                print(f"✓ Document processed")
            else:
                self.document_embeddings[document_id] = None
                
        except Exception as e:
            print(f"Error: {e}")
            self.document_chunks[document_id] = self.chunk_text(text)
            self.document_embeddings[document_id] = None
    
    def find_relevant_chunks(self, document_id: int, question: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if document_id not in self.document_chunks:
            raise ValueError(f"Document {document_id} not processed")
        
        chunks = self.document_chunks[document_id]
        
        if self.document_embeddings.get(document_id) is None:
            return self._keyword_search(chunks, question, top_k)
        
        try:
            question_embedding = self.embedding_model.encode([question], show_progress_bar=False)
            similarities = cosine_similarity(question_embedding, self.document_embeddings[document_id])[0]
            
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            return [(chunks[idx], float(similarities[idx])) for idx in top_indices]
            
        except:
            return self._keyword_search(chunks, question, top_k)
    
    def _keyword_search(self, chunks: List[str], question: str, top_k: int = 5) -> List[Tuple[str, float]]:
        question_words = set(question.lower().split())
        stop_words = {'the', 'a', 'an', 'is', 'are', 'what', 'who', 'when', 'where', 'how', 'why'}
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
        """Smart question answering with pattern matching"""
        try:
            question_lower = question.lower()
            
            # SPECIFIC QUESTION TYPE HANDLERS
            # Date/Publication questions
            if any(word in question_lower for word in ['when', 'published', 'date', 'year']):
                return self._extract_publication_info(document_id, question)
            
            # Author questions
            elif any(word in question_lower for word in ['author', 'who wrote', 'written by']):
                return self._extract_authors(document_id, question)
            
            # Abstract/Summary questions
            elif any(word in question_lower for word in ['abstract', 'summary', 'give abstract']):
                return self._extract_abstract(document_id)
            
            # Objective/Purpose questions
            elif any(word in question_lower for word in ['objective', 'purpose', 'goal', 'aim']):
                return self._extract_objective(document_id, question)
            
            # Method/Methodology questions
            elif any(word in question_lower for word in ['method', 'methodology', 'approach', 'technique']):
                return self._extract_methodology(document_id, question)
            
            # Results questions
            elif any(word in question_lower for word in ['result', 'finding', 'outcome', 'conclusion']):
                return self._extract_results(document_id, question)
            
            # DEFAULT: Use semantic search + QA model
            else:
                return self._general_qa(document_id, question)
                
        except Exception as e:
            print(f"Error: {e}")
            return "Unable to answer this question.", "", 0.0
    
    def _extract_publication_info(self, document_id: int, question: str) -> Tuple[str, str, float]:
        """Extract publication date/year"""
        full_text = ' '.join(self.document_chunks[document_id])
        
        # Look for dates in various formats
        patterns = [
            r'(?:published|released|issued|dated)(?:\s+on|\s+in|:)?\s*(\d{4}[-/]\d{2}[-/]\d{2})',
            r'(?:published|released|issued|dated)(?:\s+on|\s+in|:)?\s*([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})',
            r'(?:published|released|issued|dated)(?:\s+on|\s+in|:)?\s*([A-Z][a-z]+\s+\d{4})',
            r'(\d{4}[-/]\d{2}[-/]\d{2})\s+\[',
            r'(\d{1,2}\s+[A-Z][a-z]+\s+\d{4})',
            r'\b(20\d{2})\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, full_text)
            if matches:
                date = matches[0]
                # Find context
                pos = full_text.find(date)
                context = full_text[max(0, pos-200):min(len(full_text), pos+200)]
                return date, context, 0.90
        
        return "Publication date not found in document.", full_text[:500], 0.3
    
    def _extract_authors(self, document_id: int, question: str) -> Tuple[str, str, float]:
        """Extract author names"""
        chunks = self.find_relevant_chunks(document_id, "author", top_k=3)
        text = chunks[0][0]
        
        # Look for author patterns
        patterns = [
            r'(?:Author|Authors?|By)[:\s]+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)*[A-Z][a-z]+(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z]\.?\s*)*[A-Z][a-z]+)*)',
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+and\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'([A-Z][a-z]+,\s+[A-Z]\.(?:\s+[A-Z]\.)?)\s*(?:and|,)\s+([A-Z][a-z]+,\s+[A-Z]\.)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                authors = match.group(0)
                return authors, text[:500], 0.85
        
        # Fallback: use QA model
        return self._general_qa(document_id, question)
    
    def _extract_abstract(self, document_id: int) -> Tuple[str, str, float]:
        """Extract abstract section"""
        full_text = ' '.join(self.document_chunks[document_id])
        
        # Find abstract section
        pattern = r'(?:ABSTRACT|Abstract)[:\s]+(.*?)(?:INTRODUCTION|Introduction|KEYWORDS|Keywords|\n\n[A-Z][A-Z])'
        match = re.search(pattern, full_text, re.DOTALL | re.IGNORECASE)
        
        if match:
            abstract = match.group(1).strip()
            abstract = re.sub(r'\s+', ' ', abstract)
            if len(abstract) > 1000:
                abstract = abstract[:1000] + '...'
            return abstract, abstract, 0.95
        
        # Fallback: return first chunk
        first_chunk = self.document_chunks[document_id][0]
        return first_chunk[:800], first_chunk, 0.60
    
    def _extract_objective(self, document_id: int, question: str) -> Tuple[str, str, float]:
        """Extract objective/purpose"""
        chunks = self.find_relevant_chunks(document_id, "objective purpose goal", top_k=3)
        context = ' '.join([c for c, _ in chunks[:2]])
        
        patterns = [
            r'(?:objective|purpose|goal|aim)(?:s)?\s+(?:is|are|of\s+(?:this|the)\s+(?:study|paper|work))[\s:]+(.*?)[.;]',
            r'(?:this|the)\s+(?:study|paper|work)\s+(?:aims?|proposes?)\s+(?:to\s+)?(.*?)[.;]',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                objective = match.group(1).strip()
                return objective, context[:600], 0.85
        
        # Fallback
        return self._general_qa(document_id, question)
    
    def _extract_methodology(self, document_id: int, question: str) -> Tuple[str, str, float]:
        """Extract methodology"""
        chunks = self.find_relevant_chunks(document_id, "method methodology approach technique", top_k=3)
        context = ' '.join([c for c, _ in chunks[:2]])
        
        if self.qa_pipeline:
            try:
                result = self.qa_pipeline(
                    question=question,
                    context=context,
                    max_answer_len=250
                )
                return result['answer'], context[:700], max(result['score'], 0.65)
            except:
                pass
        
        # Extract sentences
        sentences = [s.strip() + '.' for s in re.split(r'[.!?]+', context) if len(s.strip()) > 30]
        answer = ' '.join(sentences[:3])
        return answer, context[:700], 0.60
    
    def _extract_results(self, document_id: int, question: str) -> Tuple[str, str, float]:
        """Extract results/findings"""
        chunks = self.find_relevant_chunks(document_id, "results findings outcomes conclusion", top_k=3)
        context = ' '.join([c for c, _ in chunks[:2]])
        
        if self.qa_pipeline:
            try:
                result = self.qa_pipeline(
                    question=question,
                    context=context,
                    max_answer_len=250
                )
                return result['answer'], context[:700], max(result['score'], 0.65)
            except:
                pass
        
        sentences = [s.strip() + '.' for s in re.split(r'[.!?]+', context) if len(s.strip()) > 30]
        answer = ' '.join(sentences[:3])
        return answer, context[:700], 0.60
    
    def _general_qa(self, document_id: int, question: str) -> Tuple[str, str, float]:
        """General question answering"""
        chunks = self.find_relevant_chunks(document_id, question, top_k=5)
        context = ' '.join([c for c, s in chunks[:3] if s > 0.1])
        
        if not context or len(context) < 50:
            context = chunks[0][0]
        
        if len(context) > 4000:
            context = context[:4000]
        
        if self.qa_pipeline:
            try:
                result = self.qa_pipeline(
                    question=question,
                    context=context,
                    max_answer_len=200
                )
                
                answer = result['answer'].strip()
                if len(answer) > 2 and answer != '.':
                    return answer, context[:700], max(result['score'], 0.55)
            except Exception as e:
                print(f"QA error: {e}")
        
        # Fallback: extract relevant sentences
        sentences = [s.strip() for s in re.split(r'[.!?]+', context) if len(s.strip()) > 20]
        if sentences:
            answer = '. '.join(sentences[:3]) + '.'
            return answer, context[:700], 0.55
        
        return "Could not find answer in document.", context[:500], 0.3