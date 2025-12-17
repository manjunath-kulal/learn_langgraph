import os
import json
import re
import time
from typing import Optional, Dict, Any
from functools import wraps

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from dotenv import load_dotenv
load_dotenv()

# --- 1. RATE LIMITER ---
class RateLimiter:
    def __init__(self, max_calls_per_minute=10):
        self.interval = 60.0 / max_calls_per_minute
        self.last_call_time = 0

    def wait(self):
        elapsed = time.time() - self.last_call_time
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self.last_call_time = time.time()

limiter = RateLimiter(max_calls_per_minute=15)

def rate_limit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        limiter.wait()
        return func(*args, **kwargs)
    return wrapper

# --- 2. SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "resume_index")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

try:
    vector_store = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    print("✅ FAISS Index Loaded.")
except Exception as e:
    print(f"❌ Error loading index: {e}")
    vector_store = None

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash", 
    api_key=os.environ.get("GEMINI_API_KEY"),
    temperature=0
)

# --- 3. ROBUST JSON EXTRACTOR (The Fix) ---
def extract_json(text):
    """
    Finds the first valid JSON object in a string, ignoring surrounding text.
    """
    try:
        # 1. Try standard cleaning first
        clean_text = re.sub(r"```json\s*|```", "", text).strip()
        return json.loads(clean_text)
    except json.JSONDecodeError:
        pass

    # 2. Regex search for { ... }
    # This pattern finds the first '{' and the last '}' 
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
            
    raise ValueError("No valid JSON found in response")

# --- 4. NODE FUNCTIONS ---

def retrieve_resumes(state):
    print("--- NODE: RETRIEVING RESUMES ---")
    query = state["messages"][-1].content
    docs = vector_store.similarity_search(query, k=5)
    context_text = "\n---\n".join([d.page_content for d in docs])
    return {"retrieved_docs": context_text}

@rate_limit
def check_intent_with_llm(query: str, current_candidate: Optional[Dict[str, Any]]) -> str:
    candidate_name = current_candidate.get('best_candidate_name') if current_candidate else "None"
    system_prompt = f"""You are a routing assistant.
    Current Candidate: {candidate_name}
    Classify USER QUERY into: 'FOLLOW_UP' or 'NEW_SEARCH'.
    Output ONLY the class name."""
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=query)])
    return response.content.strip().upper()

def classify_intent_node(state):
    print("--- NODE: CLASSIFYING INTENT ---")
    last_msg = state["messages"][-1].content
    candidate = state.get("selected_candidate")
    decision = check_intent_with_llm(last_msg, candidate)
    if not candidate: decision = "NEW_SEARCH"
    return {"intent": decision}

@rate_limit
def analyze_query(state):
    print("--- NODE: ANALYZING QUERY ---")
    query = state["messages"][-1].content
    system_prompt = """Analyze request. Return 'CLARIFY' if vague (no skills/role), else 'SEARCH'."""
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=query)])
    return {"is_clarification_needed": (response.content.strip().upper() == "CLARIFY")}

@rate_limit
def grade_candidates(state):
    print("--- NODE: GRADING CANDIDATES ---")
    query = state["messages"][-1].content
    docs = state["retrieved_docs"]
    
    system_prompt = f"""You are an Expert Recruitment AI.
    USER REQUIREMENT: {query}
    RESUME DATA STORE: {docs}
    
    TASK: Select best candidate.
    OUTPUT FORMAT (Strict JSON):
    {{
        "best_candidate_name": "Name",
        "match_score": 85,
        "evidence": "Brief quote...",
        "full_summary": "Summary...",
        "confidence": "HIGH" or "LOW"
    }}
    """
    
    response = llm.invoke([HumanMessage(content=system_prompt)])
    
    try:
        # USE THE NEW ROBUST EXTRACTOR
        analysis_dict = extract_json(response.content)
        
        return {
            "analysis": response.content,
            "selected_candidate": analysis_dict, 
            "confidence": analysis_dict.get("confidence", "HIGH")
        }
    except Exception as e:
        print(f"⚠️ JSON Parsing Failed: {e}")
        print(f"DEBUG: Raw content was: {response.content[:100]}...") # Print preview for debugging
        
        # Fallback that allows human to read the raw text
        return {
            "analysis": response.content,
            "confidence": "LOW",
            "selected_candidate": {
                "best_candidate_name": "Parsing Error (See Evidence)",
                "match_score": 0,
                # Store the RAW text here so the human can see it in final answer
                "evidence": f"RAW OUTPUT: {response.content}", 
                "full_summary": "The AI response could not be converted to JSON."
            }
        }

@rate_limit
def answer_follow_up_node(state):
    print("--- NODE: GENERATING FOLLOW-UP ---")
    candidate = state["selected_candidate"]
    query = state["messages"][-1].content
    context = f"Candidate: {candidate.get('best_candidate_name')}\nProfile: {candidate.get('full_summary')}\nEvidence: {candidate.get('evidence')}"
    system_prompt = f"Answer user question based ONLY on this context:\n{context}"
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=query)])
    return {"messages": [AIMessage(content=response.content)]}