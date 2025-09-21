#!/usr/bin/env python3
"""
HTTP Intent Classification Server
Simple REST API server that handles intent classification
Run with: python intent_server_http.py
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import json
import os
import sys
from typing import List, Dict, Optional, Tuple
import re
import logging

# Install required packages if not available
def install_packages():
    packages = ["fastapi", "uvicorn", "scikit-learn", "numpy", "python-dotenv"]
    for package in packages:
        try:
            if package == "scikit-learn":
                import sklearn
            elif package == "python-dotenv":
                import dotenv
            else:
                __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_packages()

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

# Import after installation
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("intent-server")

# Configuration
SERVER_HOST = os.getenv('SERVER_HOST', '0.0.0.0')
SERVER_PORT = int(os.getenv('SERVER_PORT', '8000'))
DEFAULT_THRESHOLD = float(os.getenv('INTENT_THRESHOLD_DEFAULT', '0.3'))
MAX_FEATURES = int(os.getenv('INTENT_MAX_FEATURES', '5000'))
NGRAM_MIN = int(os.getenv('INTENT_NGRAM_MIN', '1'))
NGRAM_MAX = int(os.getenv('INTENT_NGRAM_MAX', '3'))
DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

# Pydantic models for API
class QAData(BaseModel):
    qna: List[Dict]

class ClassifyRequest(BaseModel):
    query: str
    threshold: Optional[float] = None

class TopMatchesRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class IntentByIdRequest(BaseModel):
    qid: str

class IntentMatcher:
    """Intent classification using TF-IDF and cosine similarity"""
    
    def __init__(self, qa_data: List[Dict]):
        self.qa_data = qa_data
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            ngram_range=(NGRAM_MIN, NGRAM_MAX),
            max_features=MAX_FEATURES
        )
        self._prepare_vectors()
        
        # Metrics tracking
        self.metrics = {
            'total_queries': 0,
            'successful_matches': 0,
            'failed_matches': 0,
            'avg_confidence': 0.0
        }
    
    def _prepare_vectors(self):
        """Prepare TF-IDF vectors for all questions"""
        all_questions = []
        self.question_map = {}
        
        for idx, item in enumerate(self.qa_data):
            questions = item.get('q', [])
            for q in questions:
                all_questions.append(q.lower().strip())
                self.question_map[len(all_questions) - 1] = idx
        
        if all_questions:
            self.question_vectors = self.vectorizer.fit_transform(all_questions)
        else:
            self.question_vectors = None
            
        logger.info(f"Prepared vectors for {len(all_questions)} questions from {len(self.qa_data)} Q&A pairs")
    
    def find_best_match(self, user_query: str, threshold: float = None) -> Tuple[Optional[Dict], float]:
        """Find the best matching Q&A pair for the user query"""
        if threshold is None:
            threshold = DEFAULT_THRESHOLD
            
        if self.question_vectors is None or self.question_vectors.shape[0] == 0:
            return None, 0.0
        
        # Vectorize user query
        user_vector = self.vectorizer.transform([user_query.lower().strip()])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(user_vector, self.question_vectors).flatten()
        
        # Get best match
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        # Update metrics
        self.metrics['total_queries'] += 1
        if best_score >= threshold:
            self.metrics['successful_matches'] += 1
        else:
            self.metrics['failed_matches'] += 1
        
        # Update average confidence
        total = self.metrics['total_queries']
        self.metrics['avg_confidence'] = (
            (self.metrics['avg_confidence'] * (total - 1) + best_score) / total
        )
        
        if DEBUG_MODE:
            logger.debug(f"Query: '{user_query}' | Best score: {best_score:.3f} | Threshold: {threshold}")
        
        if best_score >= threshold:
            qa_idx = self.question_map[best_idx]
            return self.qa_data[qa_idx], best_score
        
        return None, best_score

    def get_top_matches(self, user_query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Get top K matching Q&A pairs"""
        if self.question_vectors is None:
            return []
        
        user_vector = self.vectorizer.transform([user_query.lower().strip()])
        similarities = cosine_similarity(user_vector, self.question_vectors).flatten()
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                qa_idx = self.question_map[idx]
                results.append((self.qa_data[qa_idx], similarities[idx]))
        
        return results

def clean_template_syntax(text: str) -> str:
    """Clean template syntax from answer text"""
    if not text:
        return "No answer available"
    
    text = re.sub(r'\{\{[^}]+\}\}', '', text)
    text = re.sub(r'\* \[([^\]]+)\]\(([^)]+)\)', r'• \1: \2', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = text.strip()
    
    return text

# Global intent matcher instance
intent_matcher: Optional[IntentMatcher] = None

# FastAPI app
app = FastAPI(
    title="Intent Classification Server",
    description="REST API for intent classification and routing",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Intent Classification Server is running",
        "status": "healthy",
        "data_loaded": intent_matcher is not None
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "data_loaded": intent_matcher is not None,
        "total_qa_pairs": len(intent_matcher.qa_data) if intent_matcher else 0,
        "config": {
            "default_threshold": DEFAULT_THRESHOLD,
            "max_features": MAX_FEATURES,
            "ngram_range": f"{NGRAM_MIN}-{NGRAM_MAX}",
            "debug_mode": DEBUG_MODE
        }
    }

@app.post("/load_data")
async def load_qa_data(data: QAData):
    """Load Q&A data for intent classification"""
    global intent_matcher
    
    try:
        qa_data = data.qna
        intent_matcher = IntentMatcher(qa_data)
        
        logger.info(f"Loaded {len(qa_data)} Q&A pairs")
        
        return {
            "status": "success",
            "message": f"Successfully loaded {len(qa_data)} Q&A pairs for intent classification",
            "total_pairs": len(qa_data),
            "total_questions": sum(len(item.get('q', [])) for item in qa_data)
        }
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise HTTPException(status_code=400, detail=f"Error loading data: {str(e)}")

@app.post("/classify")
async def classify_intent(request: ClassifyRequest):
    """Classify user query intent"""
    global intent_matcher
    
    if not intent_matcher:
        raise HTTPException(status_code=400, detail="No Q&A data loaded. Please load data first.")
    
    try:
        threshold = request.threshold if request.threshold is not None else DEFAULT_THRESHOLD
        best_match, confidence = intent_matcher.find_best_match(request.query, threshold)
        
        if best_match:
            # Found a good match
            raw_answer = best_match.get('a', 'No answer available')
            cleaned_answer = clean_template_syntax(raw_answer)
            
            result = {
                "status": "match_found",
                "confidence": confidence,
                "threshold": threshold,
                "intent": {
                    "qid": best_match.get('qid', ''),
                    "title": best_match.get('title', ''),
                    "questions": best_match.get('q', []),
                    "answer": cleaned_answer,
                    "buttons": best_match.get('r', {}).get('buttons', [])
                }
            }
        else:
            # No good match
            result = {
                "status": "no_match",
                "confidence": confidence,
                "threshold": threshold,
                "message": "No intent match found. Please try rephrasing your question."
            }
        
        return result
        
    except Exception as e:
        logger.error(f"Error classifying intent: {e}")
        raise HTTPException(status_code=500, detail=f"Error classifying intent: {str(e)}")

@app.post("/top_matches")
async def get_top_matches(request: TopMatchesRequest):
    """Get top K similar intents for a query"""
    global intent_matcher
    
    if not intent_matcher:
        raise HTTPException(status_code=400, detail="No Q&A data loaded. Please load data first.")
    
    try:
        matches = intent_matcher.get_top_matches(request.query, request.top_k)
        
        result = {
            "query": request.query,
            "top_matches": []
        }
        
        for match, score in matches:
            result["top_matches"].append({
                "confidence": score,
                "qid": match.get('qid', ''),
                "title": match.get('title', ''),
                "sample_question": match.get('q', [''])[0] if match.get('q') else '',
                "answer_preview": clean_template_syntax(match.get('a', ''))[:200] + "..." if len(match.get('a', '')) > 200 else clean_template_syntax(match.get('a', ''))
            })
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting top matches: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting top matches: {str(e)}")

@app.post("/intent_by_id")
async def get_intent_by_id(request: IntentByIdRequest):
    """Get specific intent by question ID"""
    global intent_matcher
    
    if not intent_matcher:
        raise HTTPException(status_code=400, detail="No Q&A data loaded. Please load data first.")
    
    try:
        # Find intent by QID
        for item in intent_matcher.qa_data:
            if item.get('qid') == request.qid:
                result = {
                    "found": True,
                    "intent": {
                        "qid": item.get('qid', ''),
                        "title": item.get('title', ''),
                        "questions": item.get('q', []),
                        "answer": clean_template_syntax(item.get('a', '')),
                        "buttons": item.get('r', {}).get('buttons', [])
                    }
                }
                return result
        
        # Not found
        return {
            "found": False,
            "message": f"No intent found with QID: {request.qid}"
        }
        
    except Exception as e:
        logger.error(f"Error looking up intent: {e}")
        raise HTTPException(status_code=500, detail=f"Error looking up intent: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get server statistics"""
    global intent_matcher
    
    if not intent_matcher:
        return {
            "status": "not_loaded",
            "message": "No Q&A data loaded"
        }
    
    return {
        "status": "loaded",
        "total_qa_pairs": len(intent_matcher.qa_data),
        "total_questions": sum(len(item.get('q', [])) for item in intent_matcher.qa_data),
        "metrics": intent_matcher.metrics,
        "config": {
            "default_threshold": DEFAULT_THRESHOLD,
            "max_features": MAX_FEATURES,
            "ngram_range": f"{NGRAM_MIN}-{NGRAM_MAX}",
            "debug_mode": DEBUG_MODE
        }
    }

@app.post("/reset_metrics")
async def reset_metrics():
    """Reset server metrics"""
    global intent_matcher
    
    if not intent_matcher:
        raise HTTPException(status_code=400, detail="No Q&A data loaded")
    
    intent_matcher.metrics = {
        'total_queries': 0,
        'successful_matches': 0,
        'failed_matches': 0,
        'avg_confidence': 0.0
    }
    
    return {"message": "Metrics reset successfully"}

@app.get("/qa_data")
async def get_qa_data():
    """Get all loaded Q&A data"""
    global intent_matcher
    
    if not intent_matcher:
        raise HTTPException(status_code=400, detail="No Q&A data loaded")
    
    return {
        "qna": intent_matcher.qa_data
    }

def main():
    """Run the server"""
    logger.info(f"Starting Intent Classification HTTP Server...")
    logger.info(f"Server will run on http://{SERVER_HOST}:{SERVER_PORT}")
    logger.info(f"Configuration:")
    logger.info(f"  Default threshold: {DEFAULT_THRESHOLD}")
    logger.info(f"  Max features: {MAX_FEATURES}")
    logger.info(f"  N-gram range: {NGRAM_MIN}-{NGRAM_MAX}")
    logger.info(f"  Debug mode: {DEBUG_MODE}")
    
    # Auto-load data if path is provided
    qa_data_path = os.getenv('QA_DATA_PATH', '')
    if qa_data_path and os.path.exists(qa_data_path):
        try:
            with open(qa_data_path, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
            
            if isinstance(qa_data, dict) and 'qna' in qa_data:
                qa_data = qa_data['qna']
            elif not isinstance(qa_data, list):
                qa_data = [qa_data]
            
            global intent_matcher
            intent_matcher = IntentMatcher(qa_data)
            logger.info(f"Auto-loaded {len(qa_data)} Q&A pairs from {qa_data_path}")
            
        except Exception as e:
            logger.error(f"Failed to auto-load data from {qa_data_path}: {e}")
    
    # Run server
    uvicorn.run(
        app, 
        host=SERVER_HOST, 
        port=SERVER_PORT,
        log_level="info"
    )

if __name__ == "__main__":
    main()
