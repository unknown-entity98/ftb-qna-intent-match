#!/usr/bin/env python3
"""
Improved Semantic Intent Classification Server
Uses OpenAI embeddings + configurable LLM models for classification
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import json
import os
import sys
from typing import List, Dict, Optional
import re
import logging
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import anthropic

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("semantic-intent-server")

# Configuration
SERVER_HOST = os.getenv('SERVER_HOST', '0.0.0.0')
SERVER_PORT = int(os.getenv('SERVER_PORT', '8000'))
DEFAULT_THRESHOLD = float(os.getenv('INTENT_THRESHOLD_DEFAULT', '0.75'))
DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

# Pydantic models for API
class QAData(BaseModel):
    qna: List[Dict]

class ClassifyRequest(BaseModel):
    query: str
    threshold: Optional[float] = None
    model_provider: Optional[str] = "anthropic"  # Default to anthropic since OpenAI isn't working
    model_name: Optional[str] = "claude-3-haiku-20240307"  # Safe default

class SemanticIntentMatcher:
    """Semantic intent classification using embeddings + configurable LLM"""
    
    def __init__(self, qa_data: List[Dict]):
        self.qa_data = qa_data
        self.openai_client = None
        self.anthropic_client = None
        self.embeddings = []
        self.question_texts = []
        self.qid_mapping = {}
        
        self._init_clients()
        self._build_semantic_index()
        
        # Metrics tracking
        self.metrics = {
            'total_queries': 0,
            'successful_matches': 0,
            'failed_matches': 0,
            'avg_confidence': 0.0,
            'embedding_retrievals': 0,
            'llm_validations': 0,
            'openai_calls': 0,
            'anthropic_calls': 0
        }
        
        logger.info(f"Initialized Semantic Intent Matcher with {len(qa_data)} Q&A pairs")
    
    def _init_clients(self):
        """Initialize AI clients"""
        # OpenAI client
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            try:
                self.openai_client = openai.OpenAI(api_key=openai_key)
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
        else:
            logger.warning("OPENAI_API_KEY not found in environment")
        
        # Anthropic client
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        if anthropic_key:
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
                logger.info("Anthropic client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")
        else:
            logger.warning("ANTHROPIC_API_KEY not found in environment")
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get OpenAI embedding for text"""
        if not self.openai_client:
            raise Exception("OpenAI client not available for embeddings")
        
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text.replace('\n', ' ')
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            raise e
    
    def _build_semantic_index(self):
        """Build semantic embedding index (fallback to text matching if OpenAI unavailable)"""
        if not self.openai_client:
            logger.warning("OpenAI client not available - skipping embeddings, will use text-based fallback")
            # Just prepare question texts for fallback matching
            for i, item in enumerate(self.qa_data):
                qid = item.get('qid', f'Q{i+1}')
                questions = item.get('q', [])
                title = item.get('title', '')
                
                combined_text = ' '.join(questions)
                if title:
                    combined_text += f' {title}'
                
                self.question_texts.append(combined_text.lower())
                self.qid_mapping[len(self.question_texts) - 1] = qid
            
            logger.info(f"Prepared {len(self.question_texts)} Q&A items for text-based matching (fast upload)")
            return
            
        logger.info("Building semantic embedding index (this may take a moment)...")
        
        # Extract all questions and create mapping
        for i, item in enumerate(self.qa_data):
            qid = item.get('qid', f'Q{i+1}')
            questions = item.get('q', [])
            title = item.get('title', '')
            
            # Combine questions and title for better matching
            combined_text = ' '.join(questions)
            if title:
                combined_text += f' {title}'
            
            self.question_texts.append(combined_text)
            self.qid_mapping[len(self.question_texts) - 1] = qid
            
            # Get embedding with shorter timeout for faster failure
            try:
                embedding = self._get_embedding(combined_text)
                self.embeddings.append(embedding)
                if i % 10 == 0:  # Progress indicator
                    logger.info(f"Generated embeddings for {i+1}/{len(self.qa_data)} items")
            except Exception as e:
                logger.warning(f"Failed to get embedding for QID {qid}: {e}")
                # Use zero vector as fallback
                self.embeddings.append(np.zeros(1536))
        
        if len(self.embeddings) > 0:
            self.embeddings = np.array(self.embeddings)
            logger.info(f"Built semantic index for {len(self.embeddings)} Q&A items")
        else:
            logger.warning("No embeddings generated, will use text-based fallback")
    
    def _retrieve_candidates_fallback(self, user_query: str, top_k: int = 5) -> List[Dict]:
        """Fallback candidate retrieval using text similarity when embeddings unavailable"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        try:
            # Create TF-IDF vectorizer as fallback
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1
            )
            
            # Add user query to texts for vectorization
            all_texts = self.question_texts + [user_query.lower()]
            vectors = vectorizer.fit_transform(all_texts)
            
            # Calculate similarities (last vector is user query)
            query_vector = vectors[-1]
            text_vectors = vectors[:-1]
            similarities = cosine_similarity(query_vector, text_vectors).flatten()
            
            # Get top-k candidates
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            candidates = []
            for idx in top_indices:
                similarity_score = similarities[idx]
                qid = self.qid_mapping[idx]
                
                # Find the original Q&A item
                qa_item = None
                for item in self.qa_data:
                    if item.get('qid') == qid:
                        qa_item = item
                        break
                
                if qa_item:
                    candidates.append({
                        'qid': qid,
                        'similarity': float(similarity_score),
                        'qa_item': qa_item
                    })
            
            return candidates
            
        except Exception as e:
            logger.error(f"Fallback candidate retrieval error: {e}")
            return []

    def _retrieve_candidates(self, user_query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve top-k most similar candidates using semantic embeddings or fallback"""
        self.metrics['embedding_retrievals'] += 1
        
        # Try semantic embeddings first
        if self.openai_client and len(self.embeddings) > 0:
            try:
                # Get embedding for user query
                query_embedding = self._get_embedding(user_query)
                
                # Calculate cosine similarities
                similarities = cosine_similarity([query_embedding], self.embeddings)[0]
                
                # Get top-k candidates
                top_indices = np.argsort(similarities)[::-1][:top_k]
                
                candidates = []
                for idx in top_indices:
                    similarity_score = similarities[idx]
                    qid = self.qid_mapping[idx]
                    
                    # Find the original Q&A item
                    qa_item = None
                    for item in self.qa_data:
                        if item.get('qid') == qid:
                            qa_item = item
                            break
                    
                    if qa_item:
                        candidates.append({
                            'qid': qid,
                            'similarity': float(similarity_score),
                            'qa_item': qa_item
                        })
                
                logger.debug(f"Retrieved {len(candidates)} candidates using semantic embeddings")
                return candidates
                
            except Exception as e:
                logger.warning(f"Semantic embedding retrieval failed: {e}, falling back to text similarity")
        
        # Fallback to text-based similarity
        logger.info("Using text-based fallback for candidate retrieval")
        return self._retrieve_candidates_fallback(user_query, top_k)
    
    def _llm_validate_candidates(self, user_query: str, candidates: List[Dict], 
                               model_provider: str = "openai", model_name: str = "gpt-4o-mini") -> Dict:
        """Use configurable LLM to validate and select the best candidate"""
        self.metrics['llm_validations'] += 1
        
        logger.info(f"LLM validation starting with {model_provider}/{model_name}")
        logger.info(f"Available clients - OpenAI: {self.openai_client is not None}, Anthropic: {self.anthropic_client is not None}")
        
        # Prepare candidate information for LLM
        candidates_text = ""
        for i, candidate in enumerate(candidates):
            qa_item = candidate['qa_item']
            qid = candidate['qid']
            similarity = candidate['similarity']
            questions = qa_item.get('q', [])
            title = qa_item.get('title', '')
            answer_preview = qa_item.get('a', '')[:120] + "..." if len(qa_item.get('a', '')) > 120 else qa_item.get('a', '')
            
            candidates_text += f"\nCandidate {i+1}:\n"
            candidates_text += f"- QID: {qid}\n"
            candidates_text += f"- Semantic Similarity: {similarity:.3f}\n"
            candidates_text += f"- Title: {title}\n"
            candidates_text += f"- Sample Questions: {', '.join(questions[:3])}\n"
            candidates_text += f"- Answer Preview: {answer_preview}\n"
            candidates_text += "---\n"
        
        prompt = f"""You are an expert DMV customer service intent classifier. You have been given semantically similar candidates from an embedding search. Your job is to determine if any truly match the user's intent.

USER QUERY: "{user_query}"

SEMANTIC CANDIDATES:
{candidates_text}

INSTRUCTIONS:
1. Understand what the user is actually trying to accomplish
2. Evaluate each candidate for true semantic relevance
3. Consider context, intent, and implied meaning
4. If there's a genuine match, return that QID with appropriate confidence
5. If no candidate truly matches the user's intent, return null
6. Be conservative - only high confidence matches should pass

CRITICAL: Respond with ONLY valid JSON, no additional text before or after.

{{
    "selected_qid": "QID_FROM_CANDIDATES or null",
    "confidence": 0.85,
    "reasoning": "Detailed explanation of why this candidate matches or why none match",
    "user_intent": "Clear description of what the user wants to accomplish"
}}

Return only the JSON object above, nothing else."""

        try:
            if model_provider.lower() == "anthropic" and self.anthropic_client:
                logger.info("Using Anthropic client for validation")
                self.metrics['anthropic_calls'] += 1
                response = self.anthropic_client.messages.create(
                    model=model_name,
                    max_tokens=300,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content[0].text.strip()
                logger.info(f"Anthropic response received: {content[:100]}...")
                
            elif model_provider.lower() == "openai" and self.openai_client:
                logger.info("Using OpenAI client for validation")
                self.metrics['openai_calls'] += 1
                response = self.openai_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert intent classification system. Always respond with valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=300
                )
                content = response.choices[0].message.content.strip()
                logger.info(f"OpenAI response received: {content[:100]}...")
                
            else:
                logger.error(f"No available client for {model_provider}")
                return {
                    "selected_qid": None,
                    "confidence": 0.0,
                    "reasoning": f"No {model_provider} client available",
                    "user_intent": "Unable to determine - no client"
                }
            
            # Extract JSON from response - more robust parsing
            logger.info(f"Raw LLM response: {content}")
            
            # Try to find JSON in the response
            json_content = content
            
            # Remove markdown code blocks
            if content.startswith('```json'):
                json_content = content[7:]
                if json_content.endswith('```'):
                    json_content = json_content[:-3]
            elif content.startswith('```'):
                json_content = content[3:]
                if json_content.endswith('```'):
                    json_content = json_content[:-3]
            
            # Find JSON object boundaries
            json_content = json_content.strip()
            
            # Look for the first { and last } to extract just the JSON part
            start_idx = json_content.find('{')
            end_idx = json_content.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_content = json_content[start_idx:end_idx+1]
            
            logger.info(f"Extracted JSON content: {json_content}")
            
            try:
                parsed_result = json.loads(json_content)
            except json.JSONDecodeError as parse_error:
                logger.error(f"JSON parsing still failed: {parse_error}")
                logger.error(f"Attempted to parse: {json_content}")
                
                # Fallback: try to extract fields manually if JSON is malformed
                try:
                    # Look for key patterns in the text
                    selected_qid = None
                    confidence = 0.0
                    reasoning = "JSON parsing failed, but extracted from text"
                    user_intent = "Unable to determine"
                    
                    # Try to extract QID
                    if '"selected_qid"' in content:
                        qid_match = re.search(r'"selected_qid":\s*"([^"]*)"', content)
                        if qid_match:
                            selected_qid = qid_match.group(1)
                            if selected_qid.lower() == "null":
                                selected_qid = None
                    
                    # Try to extract confidence
                    if '"confidence"' in content:
                        conf_match = re.search(r'"confidence":\s*([0-9.]+)', content)
                        if conf_match:
                            confidence = float(conf_match.group(1))
                    
                    # Try to extract reasoning
                    if '"reasoning"' in content:
                        reason_match = re.search(r'"reasoning":\s*"([^"]*)"', content)
                        if reason_match:
                            reasoning = reason_match.group(1)
                    
                    return {
                        "selected_qid": selected_qid,
                        "confidence": confidence,
                        "reasoning": f"Manual extraction: {reasoning}",
                        "user_intent": user_intent
                    }
                    
                except Exception as fallback_error:
                    logger.error(f"Fallback extraction failed: {fallback_error}")
                    return {
                        "selected_qid": None,
                        "confidence": 0.0,
                        "reasoning": f"Complete JSON parsing failure: {str(parse_error)}",
                        "user_intent": "Unable to determine"
                    }
            
            logger.info(f"Successfully parsed LLM result: {parsed_result}")
            return parsed_result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Raw content was: {content}")
            return {
                "selected_qid": None,
                "confidence": 0.0,
                "reasoning": f"JSON parsing failed: {str(e)}",
                "user_intent": "Unable to determine - parsing error"
            }
        except Exception as e:
            logger.error(f"LLM validation error: {e}")
            return {
                "selected_qid": None,
                "confidence": 0.0,
                "reasoning": f"LLM validation failed: {str(e)}",
                "user_intent": "Unable to determine - LLM error"
            }
    
    def match_intent(self, user_query: str, threshold: float = None, 
                    model_provider: str = "openai", model_name: str = "gpt-4o-mini") -> Dict:
        """Semantic intent matching: embedding retrieval + LLM validation"""
        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        
        self.metrics['total_queries'] += 1
        
        logger.info(f"=== INTENT MATCHING DEBUG ===")
        logger.info(f"Query: '{user_query}'")
        logger.info(f"Requested: {model_provider}/{model_name}")
        logger.info(f"Threshold: {threshold}")
        logger.info(f"Available clients - OpenAI: {self.openai_client is not None}, Anthropic: {self.anthropic_client is not None}")
        
        # Check if requested provider is available
        if model_provider.lower() == "openai" and not self.openai_client:
            logger.warning("OpenAI requested but not available, falling back to Anthropic")
            if self.anthropic_client:
                model_provider = "anthropic"
                model_name = "claude-3-haiku-20240307"  # Safe default
            else:
                logger.error("No AI providers available at all!")
                self.metrics['failed_matches'] += 1
                return {
                    "status": "error",
                    "confidence": 0.0,
                    "message": "No AI providers available",
                    "model_used": "none"
                }
        
        if model_provider.lower() == "anthropic" and not self.anthropic_client:
            logger.warning("Anthropic requested but not available, falling back to OpenAI")
            if self.openai_client:
                model_provider = "openai"
                model_name = "gpt-4o-mini"  # Safe default
            else:
                logger.error("No AI providers available at all!")
                self.metrics['failed_matches'] += 1
                return {
                    "status": "error",
                    "confidence": 0.0,
                    "message": "No AI providers available",
                    "model_used": "none"
                }
        
        logger.info(f"Using: {model_provider}/{model_name}")
        
        try:
            # Step 1: Candidate retrieval (embedding or fallback)
            logger.info("Step 1: Retrieving candidates...")
            candidates = self._retrieve_candidates(user_query, top_k=5)
            logger.info(f"Found {len(candidates)} candidates")
            
            if not candidates:
                logger.warning("No candidates found!")
                self.metrics['failed_matches'] += 1
                return {
                    "status": "no_candidates",
                    "confidence": 0.0,
                    "message": "No similar candidates found",
                    "model_used": f"{model_provider}/{model_name}"
                }
            
            # Log candidate info
            for i, candidate in enumerate(candidates):
                logger.info(f"Candidate {i+1}: {candidate['qid']} (similarity: {candidate['similarity']:.3f})")
            
            # Step 2: LLM validation with available model
            logger.info("Step 2: LLM validation...")
            validation_result = self._llm_validate_candidates(
                user_query, candidates, model_provider, model_name
            )
            logger.info(f"LLM validation complete: {validation_result}")
            
            selected_qid = validation_result.get('selected_qid')
            confidence = float(validation_result.get('confidence', 0.0))
            reasoning = validation_result.get('reasoning', '')
            user_intent = validation_result.get('user_intent', '')
            
            logger.info(f"Selected QID: {selected_qid}, Confidence: {confidence}, Threshold: {threshold}")
            
            # Update average confidence
            total = self.metrics['total_queries']
            self.metrics['avg_confidence'] = (
                (self.metrics['avg_confidence'] * (total - 1) + confidence) / total
            )
            
            if selected_qid and confidence >= threshold:
                logger.info("Match found and passes threshold!")
                # Find the selected Q&A item
                selected_qa = None
                for candidate in candidates:
                    if candidate['qid'] == selected_qid:
                        selected_qa = candidate['qa_item']
                        break
                
                if selected_qa:
                    self.metrics['successful_matches'] += 1
                    return {
                        "status": "match_found",
                        "qid": selected_qid,
                        "confidence": confidence,
                        "user_intent": user_intent,
                        "reasoning": reasoning,
                        "qa_item": selected_qa,
                        "method": "hybrid_retrieval_llm",
                        "model_used": f"{model_provider}/{model_name}",
                        "candidates_considered": [c['qid'] for c in candidates],
                        "retrieval_method": "semantic" if self.openai_client and len(self.embeddings) > 0 else "text_fallback"
                    }
                else:
                    logger.warning(f"Selected QID {selected_qid} not found in candidates")
            else:
                logger.info(f"No match: QID={selected_qid}, confidence={confidence} < threshold={threshold}")
            
            # No good match found
            self.metrics['failed_matches'] += 1
            return {
                "status": "no_match",
                "confidence": confidence,
                "user_intent": user_intent,
                "reasoning": reasoning,
                "message": "No sufficiently confident intent match found",
                "model_used": f"{model_provider}/{model_name}",
                "candidates_considered": [c['qid'] for c in candidates],
                "retrieval_method": "semantic" if self.openai_client and len(self.embeddings) > 0 else "text_fallback"
            }
                
        except Exception as e:
            logger.error(f"Intent matching error: {e}")
            logger.error(f"Exception type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.metrics['failed_matches'] += 1
            return {
                "status": "error", 
                "message": f"Error processing query: {str(e)}",
                "confidence": 0.0,
                "model_used": f"{model_provider}/{model_name}"
            }

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
intent_matcher: Optional[SemanticIntentMatcher] = None

# FastAPI app
app = FastAPI(
    title="Semantic Intent Classification Server",
    description="Semantic intent classification using embeddings + configurable LLM validation",
    version="4.0.0"
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
        "message": "Semantic Intent Classification Server is running",
        "status": "healthy",
        "data_loaded": intent_matcher is not None,
        "method": "semantic_embedding_llm"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    clients_status = {
        "openai_available": intent_matcher.openai_client is not None if intent_matcher else False,
        "anthropic_available": intent_matcher.anthropic_client is not None if intent_matcher else False
    }
    
    return {
        "status": "healthy",
        "data_loaded": intent_matcher is not None,
        "total_qa_pairs": len(intent_matcher.qa_data) if intent_matcher else 0,
        "method": "semantic_embedding_llm",
        "clients": clients_status,
        "config": {
            "default_threshold": DEFAULT_THRESHOLD,
            "debug_mode": DEBUG_MODE
        }
    }

@app.post("/load_data")
async def load_qa_data(data: QAData):
    """Load Q&A data for semantic intent classification"""
    global intent_matcher
    
    try:
        qa_data = data.qna
        intent_matcher = SemanticIntentMatcher(qa_data)
        
        logger.info(f"Loaded {len(qa_data)} Q&A pairs for semantic intent matching")
        
        return {
            "status": "success",
            "message": f"Successfully loaded {len(qa_data)} Q&A pairs for semantic intent classification",
            "total_pairs": len(qa_data),
            "total_questions": sum(len(item.get('q', [])) for item in qa_data),
            "method": "semantic_embedding_llm",
            "openai_available": intent_matcher.openai_client is not None,
            "anthropic_available": intent_matcher.anthropic_client is not None
        }
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise HTTPException(status_code=400, detail=f"Error loading data: {str(e)}")

@app.post("/classify")
async def classify_intent(request: ClassifyRequest):
    """Classify user query intent using semantic approach with configurable models"""
    global intent_matcher
    
    if not intent_matcher:
        raise HTTPException(status_code=400, detail="No Q&A data loaded. Please load data first.")
    
    try:
        threshold = request.threshold if request.threshold is not None else DEFAULT_THRESHOLD
        model_provider = request.model_provider or "openai"
        model_name = request.model_name or "gpt-4o-mini"
        
        result = intent_matcher.match_intent(
            request.query, threshold, model_provider, model_name
        )
        
        if result["status"] == "match_found":
            # Found a good match
            qa_item = result["qa_item"]
            raw_answer = qa_item.get('a', 'No answer available')
            cleaned_answer = clean_template_syntax(raw_answer)
            
            return {
                "status": "match_found",
                "confidence": result["confidence"],
                "threshold": threshold,
                "user_intent": result["user_intent"],
                "reasoning": result["reasoning"],
                "method": "semantic_embedding_llm",
                "model_used": result["model_used"],
                "candidates_considered": result.get("candidates_considered", []),
                "intent": {
                    "qid": qa_item.get('qid', ''),
                    "title": qa_item.get('title', ''),
                    "questions": qa_item.get('q', []),
                    "answer": cleaned_answer,
                    "buttons": qa_item.get('r', {}).get('buttons', [])
                }
            }
        else:
            # No good match or error
            return {
                "status": result["status"],
                "confidence": result.get("confidence", 0.0),
                "threshold": threshold,
                "user_intent": result.get("user_intent", ""),
                "reasoning": result.get("reasoning", ""),
                "method": "semantic_embedding_llm",
                "model_used": result.get("model_used", f"{model_provider}/{model_name}"),
                "message": result.get("message", "No intent match found"),
                "candidates_considered": result.get("candidates_considered", [])
            }
        
    except Exception as e:
        logger.error(f"Error classifying intent: {e}")
        raise HTTPException(status_code=500, detail=f"Error classifying intent: {str(e)}")

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
        "method": "semantic_embedding_llm",
        "config": {
            "default_threshold": DEFAULT_THRESHOLD,
            "debug_mode": DEBUG_MODE
        }
    }

def main():
    """Run the server"""
    logger.info(f"Starting Semantic Intent Classification Server...")
    logger.info(f"Server will run on http://{SERVER_HOST}:{SERVER_PORT}")
    logger.info(f"Method: Semantic Embeddings + Configurable LLM Validation")
    logger.info(f"Default threshold: {DEFAULT_THRESHOLD}")
    
    # Check for API keys
    openai_key = os.getenv('OPENAI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    
    if openai_key:
        logger.info("OpenAI API key found")
    else:
        logger.warning("OPENAI_API_KEY not found in environment")
    
    if anthropic_key:
        logger.info("Anthropic API key found")
    else:
        logger.warning("ANTHROPIC_API_KEY not found in environment")
    
    if not openai_key and not anthropic_key:
        logger.error("At least one API key (OpenAI or Anthropic) is required!")
        sys.exit(1)
    
    # Run server
    uvicorn.run(
        app, 
        host=SERVER_HOST, 
        port=SERVER_PORT,
        log_level="info"
    )

if __name__ == "__main__":
    main()