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
    classify_intent_node, 
    answer_follow_up_node,
    fetch_candidate_details 
)

# --- 1. DEFINE STATE ---
class RecruitmentState(TypedDict):
    # 'messages' uses operator.add to append history instead of overwriting
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

# --- 3. NODES ---
def ask_clarification_node(state):
    return {"messages": [AIMessage(content="I need more details. What tech stack or seniority level are you looking for?")]}

def human_review_node(state):
    # This runs immediately AFTER the human types "ok" and resumes execution.
    return {"messages": [AIMessage(content="✅ Human verified. Proceeding with this candidate...")]}

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
workflow.add_node("fetch_candidate_details", fetch_candidate_details)
workflow.add_node("answer_follow_up", answer_follow_up_node)
workflow.add_node("analyze_query", analyze_query)
workflow.add_node("ask_clarification", ask_clarification_node)
workflow.add_node("retrieve_resumes", retrieve_resumes)
workflow.add_node("grade_candidates", grade_candidates)
workflow.add_node("human_review", human_review_node)
workflow.add_node("final_answer", final_answer_node)

# Set Entry Point
workflow.set_entry_point("classify_intent")

# Edge 1: Intent Routing
workflow.add_conditional_edges("classify_intent", route_by_intent, 
    {"FOLLOW_UP": "fetch_candidate_details", "NEW_SEARCH": "analyze_query"})

# Edge 2: Deep Dive Flow
workflow.add_edge("fetch_candidate_details", "answer_follow_up")

# Edge 3: Standard Search Flow
workflow.add_conditional_edges("analyze_query", decide_next_step, 
    {"ask_clarification": "ask_clarification", "retrieve_resumes": "retrieve_resumes"})

workflow.add_edge("retrieve_resumes", "grade_candidates")

# Edge 4: Confidence Check (The HITL Fork)
workflow.add_conditional_edges("grade_candidates", check_confidence, 
    {"human_review": "human_review", "final_answer": "final_answer"})

# 🔥 CRITICAL FIX: Connect Human Review -> Final Answer
# This ensures that after you say "ok", the agent actually SHOWS you the candidate.
workflow.add_edge("human_review", "final_answer")

# Terminals
workflow.add_edge("answer_follow_up", END)
workflow.add_edge("ask_clarification", END)
workflow.add_edge("final_answer", END)

# --- 5. COMPILE WITH PERSISTENCE & INTERRUPT ---
checkpointer = MemorySaver()

app = workflow.compile(
    checkpointer=checkpointer, 
    interrupt_before=["human_review"] # The graph PAUSES right before entering this node
)

# --- 6. RUN APP (INTERACTIVE LOOP) ---
if __name__ == "__main__":
    print("🤖 HITL Recruitment Agent Ready. (Type 'quit' to exit)")
    
    # Config is REQUIRED for persistence to work (identifies the session)
    config = {"configurable": {"thread_id": "session_1"}}

    while True:
        try:
            # 1. CHECK INTERRUPT STATE (Are we paused?)
            current_state = app.get_state(config)
            
            # If the next node is 'human_review', the graph is currently FROZEN.
            if current_state.next and current_state.next[0] == "human_review":
                print("\n⚠️  LOW CONFIDENCE DETECTED.")
                print("   The agent is unsure. It has paused for your review.")
                user_action = input("   Type 'ok' to approve, or 'override' to reject: ")
                
                if user_action.lower() == "ok":
                    print("👍 Approving...")
                    
                    # STATE UPDATE: We manually override the confidence in the agent's memory
                    app.update_state(config, {"confidence": "HIGH"}) 
                    
                    # RESUME: Passing None tells LangGraph to "continue execution"
                    for event in app.stream(None, config=config):
                        for key, value in event.items():
                             if "messages" in value: print(f"🤖 Agent: {value['messages'][-1].content}")
                             
                else:
                    print("❌ Rejected. Resetting search.")
                    # In a real app, you might clear state here. 
                    # For now, we just loop back to let the user type a new query.
                    pass
                
                continue # Skip the normal input prompt, loop back to start

            # 2. NORMAL CHAT LOOP
            user_input = input("\nHR User: ")
            if user_input.lower() in ["quit", "exit"]: break
            
            inputs = {"messages": [HumanMessage(content=user_input)]}
            
            for event in app.stream(inputs, config=config):
                for key, value in event.items():
                    if key == "human_review":
                        # This won't print because we interrupt BEFORE it runs!
                        pass 
                    elif key in ["final_answer", "answer_follow_up", "ask_clarification"]:
                        print(f"\n🤖 Agent: {value['messages'][-1].content}")
                    else:
                        print(f"   ↳ {key}...")

        except KeyboardInterrupt:
            print("\nExiting...")
            break