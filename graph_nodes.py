import os
import json
import re
import time
import logging
from typing import Optional, Dict, Any
from functools import wraps

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from dotenv import load_dotenv
load_dotenv()

# --- 1. LOGGING CONFIGURATION (TRUE LOGGING) ---
# This sets up a logger that writes to both 'agent.log' and the terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("agent.log", mode='a'), # Append mode
        logging.StreamHandler() # Console output
    ]
)
logger = logging.getLogger("RecruitmentAgent")

# --- 2. PERSISTENT RATE LIMITER ---
class PersistentRateLimiter:
    """
    A Rate Limiter that remembers usage across script restarts 
    by saving the last call timestamp to a local file.
    """
    def __init__(self, state_file=".limiter_state", max_calls_per_min=10):
        self.state_file = os.path.join(os.path.dirname(__file__), state_file)
        self.interval = 60.0 / max_calls_per_min

    def wait_for_token(self):
        last_call = 0.0
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    content = f.read().strip()
                    if content:
                        last_call = float(content)
            except Exception:
                pass 

        now = time.time()
        elapsed = now - last_call
        
        if elapsed < self.interval:
            sleep_time = self.interval - elapsed
            logger.warning(f"⏳ Rate Limit (Proactive): Pausing for {sleep_time:.2f}s...")
            time.sleep(sleep_time)

        try:
            with open(self.state_file, "w") as f:
                f.write(str(time.time()))
        except Exception:
            pass

limiter = PersistentRateLimiter(max_calls_per_min=10)

def safe_llm_call(func):
    """
    Hybrid Decorator: Proactive Wait + Reactive Retry
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Proactive Step
        limiter.wait_for_token() 
        
        # Reactive Step
        retries = 3
        base_delay = 30 
        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    logger.warning(f"🛑 API QUOTA HIT (Attempt {attempt+1}/{retries}). Sleeping {base_delay}s...")
                    time.sleep(base_delay)
                    base_delay += 10 
                    limiter.wait_for_token() # Update state after wake
                else:
                    logger.error(f"❌ LLM Call Failed: {e}")
                    raise e
        
        logger.critical("Failed after max retries due to API Quota.")
        raise Exception("Failed after max retries due to API Quota.")
    return wrapper

# --- 3. SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "resume_index")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

try:
    vector_store = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    logger.info("✅ FAISS Index Loaded Successfully.")
except Exception as e:
    logger.critical(f"❌ Error loading index: {e}")
    vector_store = None

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash", 
    api_key=os.environ.get("GEMINI_API_KEY"),
    temperature=0
)

# --- 4. ROBUST JSON EXTRACTOR ---
def extract_json(text):
    try:
        clean_text = re.sub(r"```json\s*|```", "", text).strip()
        return json.loads(clean_text)
    except json.JSONDecodeError: pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try: return json.loads(match.group(0))
        except: pass
    logger.error("JSON Extraction Failed: No valid JSON found in response.")
    raise ValueError("No valid JSON found in response")

# --- 5. NODE FUNCTIONS (With True Logging) ---

def retrieve_resumes(state):
    logger.info("--- NODE ENTRY: retrieve_resumes (Broad Search) ---")
    query = state["messages"][-1].content
    docs = vector_store.similarity_search(query, k=5)
    context_text = "\n---\n".join([d.page_content for d in docs])
    return {"retrieved_docs": context_text}

def fetch_candidate_details(state):
    logger.info("--- NODE ENTRY: fetch_candidate_details (Deep Dive) ---")
    candidate = state["selected_candidate"]
    name = candidate.get("best_candidate_name", "")
    docs = vector_store.similarity_search(name, k=10)
    context_text = "\n---\n".join([d.page_content for d in docs])
    return {"retrieved_docs": context_text}

@safe_llm_call
def check_intent_with_llm(query: str, current_candidate: Optional[Dict[str, Any]]) -> str:
    logger.info(f"LLM CALL: check_intent_with_llm | Query: {query[:50]}...")
    candidate_name = current_candidate.get('best_candidate_name') if current_candidate else "None"
    system_prompt = f"""You are a routing assistant.
    Current Candidate: {candidate_name}
    Classify USER QUERY into: 'FOLLOW_UP' or 'NEW_SEARCH'.
    Output ONLY the class name."""
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=query)])
    return response.content.strip().upper()

def classify_intent_node(state):
    logger.info("--- NODE ENTRY: classify_intent_node ---")
    last_msg = state["messages"][-1].content
    candidate = state.get("selected_candidate")
    decision = check_intent_with_llm(last_msg, candidate)
    if not candidate: decision = "NEW_SEARCH"
    logger.info(f"Intent Decision: {decision}")
    return {"intent": decision}

@safe_llm_call
def analyze_query(state):
    logger.info("--- NODE ENTRY: analyze_query ---")
    logger.info("LLM CALL: analyze_query")
    query = state["messages"][-1].content
    system_prompt = """Analyze request. Return 'CLARIFY' if vague (no skills/role), else 'SEARCH'."""
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=query)])
    return {"is_clarification_needed": (response.content.strip().upper() == "CLARIFY")}

@safe_llm_call
def grade_candidates(state):
    logger.info("--- NODE ENTRY: grade_candidates ---")
    logger.info("LLM CALL: grade_candidates")
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
        analysis_dict = extract_json(response.content)
        return {
            "analysis": response.content,
            "selected_candidate": analysis_dict, 
            "confidence": analysis_dict.get("confidence", "HIGH")
        }
    except Exception as e:
        logger.error(f"JSON Parsing Failed: {e}")
        logger.debug(f"Raw Content: {response.content[:200]}...") # Debug level for raw content
        return {
            "analysis": response.content,
            "confidence": "LOW",
            "selected_candidate": {
                "best_candidate_name": "Parsing Error",
                "match_score": 0,
                "evidence": f"RAW OUTPUT: {response.content}", 
                "full_summary": "The AI response could not be converted to JSON."
            }
        }

@safe_llm_call
def answer_follow_up_node(state):
    logger.info("--- NODE ENTRY: answer_follow_up_node ---")
    logger.info("LLM CALL: answer_follow_up_node")
    candidate = state["selected_candidate"]
    query = state["messages"][-1].content
    full_context = state["retrieved_docs"]
    
    system_prompt = f"""You are a helpful HR assistant.
    You are answering a question about {candidate.get('best_candidate_name')}.
    FULL RESUME CONTEXT:
    {full_context}
    USER QUESTION: {query}
    Answer specific details (Education, Skills, Hobbies) found in the context.
    """
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=query)])
    return {"messages": [AIMessage(content=response.content)]}