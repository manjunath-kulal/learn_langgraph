import operator
from typing import Annotated, List, TypedDict, Literal, Optional, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver 

# Import nodes
from graph_nodes import (
    analyze_query, 
    retrieve_resumes, 
    grade_candidates, 
    classify_intent_node, # We will define the wrapper in main.py or import it
    answer_follow_up_node
)

# --- 1. DEFINE STATE ---
class RecruitmentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    retrieved_docs: str
    selected_candidate: Optional[Dict[str, Any]] 
    intent: Literal["NEW_SEARCH", "FOLLOW_UP"]
    is_clarification_needed: bool
    analysis: str
    confidence: str

# --- 2. ROUTING LOGIC ---
def route_by_intent(state):
    return state["intent"]

def decide_next_step(state):
    if state["is_clarification_needed"]:
        return "ask_clarification"
    return "retrieve_resumes"

def check_confidence(state):
    if state["confidence"] == "LOW":
        return "human_review"
    return "final_answer"

# --- 3. TERMINAL NODES ---
def ask_clarification_node(state):
    return {"messages": [AIMessage(content="I need more details. What tech stack or seniority level are you looking for?")]}

def human_review_node(state):
    return {"messages": [AIMessage(content="⚠️ LOW CONFIDENCE: Please review the analysis manually.")]}

def final_answer_node(state):
    candidate = state["selected_candidate"]
    text = f"✅ Best Match: {candidate.get('best_candidate_name')}\n"
    text += f"📊 Score: {candidate.get('match_score')}/100\n"
    text += f"💡 Evidence: {candidate.get('evidence')}"
    return {"messages": [AIMessage(content=text)]}

# --- 4. BUILD THE GRAPH ---
workflow = StateGraph(RecruitmentState)

# Add Nodes
workflow.add_node("classify_intent", classify_intent_node)
workflow.add_node("answer_follow_up", answer_follow_up_node)
workflow.add_node("analyze_query", analyze_query)
workflow.add_node("ask_clarification", ask_clarification_node)
workflow.add_node("retrieve_resumes", retrieve_resumes)
workflow.add_node("grade_candidates", grade_candidates)
workflow.add_node("human_review", human_review_node)
workflow.add_node("final_answer", final_answer_node)

# Set Entry Point
workflow.set_entry_point("classify_intent")

# Edges
workflow.add_conditional_edges(
    "classify_intent",
    route_by_intent,
    {
        "FOLLOW_UP": "answer_follow_up",
        "NEW_SEARCH": "analyze_query"
    }
)

workflow.add_conditional_edges(
    "analyze_query",
    decide_next_step,
    {
        "ask_clarification": "ask_clarification",
        "retrieve_resumes": "retrieve_resumes"
    }
)

workflow.add_edge("retrieve_resumes", "grade_candidates")

workflow.add_conditional_edges(
    "grade_candidates",
    check_confidence,
    {
        "human_review": "human_review",
        "final_answer": "final_answer"
    }
)

workflow.add_edge("answer_follow_up", END)
workflow.add_edge("ask_clarification", END)
workflow.add_edge("human_review", END)
workflow.add_edge("final_answer", END)

# --- 5. COMPILE WITH CHECKPOINTER (Your Fix) ---
# MemorySaver acts as the InMemorySaver to persist state across turns
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# --- 6. RUN APP ---
if __name__ == "__main__":
    print("🤖 Stateful Recruitment Agent Ready. (Type 'quit' to exit)")
    
    # We use a static thread_id to simulate a continuous session
    config = {"configurable": {"thread_id": "session_1"}}

    while True:
        try:
            user_input = input("\nHR User: ")
            if user_input.lower() in ["quit", "exit"]:
                break
            
            # The checkpointer will now automatically load the state from 'session_1'
            inputs = {"messages": [HumanMessage(content=user_input)]}
            
            for event in app.stream(inputs, config=config):
                for key, value in event.items():
                    if key in ["final_answer", "answer_follow_up", "ask_clarification"]:
                        print(f"\n🤖 Agent: {value['messages'][-1].content}")
                    elif key == "human_review":
                        print(f"\n⚠️ {value['messages'][-1].content}")
                    else:
                        print(f"   ↳ {key}...")
                        
        except KeyboardInterrupt:
            print("\nExiting...")
            break