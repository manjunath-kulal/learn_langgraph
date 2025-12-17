import os
import json
import re
from typing import Optional, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from dotenv import load_dotenv
load_dotenv()

# --- 1. SETUP & LOAD MEMORY ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "resume_index")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

try:
    vector_store = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    print("✅ FAISS Index Loaded.")
except Exception as e:
    print(f"❌ Error loading index: {e}")
    vector_store = None

# Initialize the Brain (Gemini 2.5 Flash) - STRICTLY PRESERVED
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    api_key=os.environ.get("GEMINI_API_KEY"),
    temperature=0
)

# --- HELPER: CLEAN JSON ---
def clean_json_string(json_str):
    return re.sub(r"```json\s*|```", "", json_str).strip()

# --- 2. NODE FUNCTIONS ---

def retrieve_resumes(state):
    print("--- NODE: RETRIEVING RESUMES ---")
    query = state["messages"][-1].content
    docs = vector_store.similarity_search(query, k=5)
    context_text = "\n---\n".join([d.page_content for d in docs])
    return {"retrieved_docs": context_text}

def check_intent_with_llm(query: str, current_candidate: Optional[Dict[str, Any]]) -> str:
    """
    Uses Gemini to strictly classify if the user is following up or starting fresh.
    """
    candidate_context = f"Current Candidate Context: {current_candidate.get('best_candidate_name')}" if current_candidate else "No candidate selected yet."
    
    system_prompt = f"""You are a routing assistant.
    {candidate_context}
    
    Classify the USER QUERY into one of two intents:
    1. 'FOLLOW_UP': The user is asking a question about the current candidate (e.g., "is he indian?", "tell me more", "what about education?", "send email").
    2. 'NEW_SEARCH': The user is asking for a different role or candidate (e.g., "find a java dev", "search for something else").
    
    Output ONLY the word 'FOLLOW_UP' or 'NEW_SEARCH'."""
    
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=query)])
    return response.content.strip().upper()

def analyze_query(state):
    print("--- NODE: ANALYZING QUERY ---")
    query = state["messages"][-1].content
    
    system_prompt = """You are a senior technical recruiter. 
    Analyze the user's request. 
    1. If they ask for a general role (e.g. "Find me a dev") without specific skills/stack, return 'CLARIFY'.
    2. If they provide specific details (e.g. "Python dev with AWS"), return 'SEARCH'.
    
    Output ONLY the word 'CLARIFY' or 'SEARCH'."""
    
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=query)])
    return {"is_clarification_needed": (response.content.strip().upper() == "CLARIFY")}

def grade_candidates(state):
    print("--- NODE: GRADING CANDIDATES ---")
    query = state["messages"][-1].content
    docs = state["retrieved_docs"]
    
    system_prompt = f"""You are an Expert Recruitment AI.
    USER REQUIREMENT: {query}
    RESUME DATA STORE: {docs}
    
    TASK:
    1. Select the best candidate.
    2. OUTPUT FORMAT (Strict JSON):
    {{
        "best_candidate_name": "Name",
        "match_score": 85,
        "evidence": "Brief quote from text supporting the match...",
        "full_summary": "A 2-3 sentence summary of the candidate's profile.",
        "confidence": "HIGH" or "LOW"
    }}
    """
    
    response = llm.invoke([HumanMessage(content=system_prompt)])
    
    try:
        clean_response = clean_json_string(response.content)
        analysis_dict = json.loads(clean_response)
        return {
            "analysis": response.content,
            "selected_candidate": analysis_dict, 
            "confidence": analysis_dict.get("confidence", "HIGH")
        }
    except Exception as e:
        print(f"⚠️ JSON Parsing Failed: {e}")
        return {"analysis": response.content, "confidence": "LOW"}

def generate_followup_answer(state):
    """
    Uses the LLM to answer questions about the specific candidate using the resume context.
    """
    print("--- NODE: GENERATING FOLLOW-UP ---")
    candidate = state["selected_candidate"]
    query = state["messages"][-1].content
    
    # We feed the FULL summary/evidence we already have to the LLM
    context = f"Candidate: {candidate.get('best_candidate_name')}\nProfile: {candidate.get('full_summary')}\nEvidence: {candidate.get('evidence')}"
    
    system_prompt = f"""You are a helpful HR assistant.
    You are answering a question about a specific candidate based ONLY on the summary below.
    
    CONTEXT:
    {context}
    
    USER QUESTION: {query}
    
    Answer clearly. If the information is not in the context, say "I don't see that mentioned in the resume summary."
    """
    
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=query)])
    return response.content