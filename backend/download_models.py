"""
Script to download NLP models before starting the server.
Run this once: python download_models.py
"""

import os
os.environ['TRANSFORMERS_CACHE'] = './models_cache'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = './models_cache'

print("=" * 60)
print("Downloading NLP Models - This may take 5-10 minutes")
print("=" * 60)

# Download NLTK data
print("\n1. Downloading NLTK data...")
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print("✓ NLTK data downloaded")
except Exception as e:
    print(f"✗ NLTK download failed: {e}")

# Download transformers model
print("\n2. Downloading Question Answering model...")
try:
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    model_name = "distilbert-base-cased-distilled-squad"
    print(f"   Downloading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    print("✓ QA model downloaded")
except Exception as e:
    print(f"✗ QA model download failed: {e}")

# Download sentence transformer
print("\n3. Downloading Sentence Embedding model...")
try:
    from sentence_transformers import SentenceTransformer
    model_name = "all-MiniLM-L6-v2"
    print(f"   Downloading {model_name}...")
    model = SentenceTransformer(model_name)
    print("✓ Embedding model downloaded")
except Exception as e:
    print(f"✗ Embedding model download failed: {e}")

print("\n" + "=" * 60)
print("✓ All models downloaded successfully!")
print("You can now start the server with:")
print("python -m uvicorn app.main:app --reload")
print("=" * 60)