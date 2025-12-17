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

# --- 3. NODES ---
def ask_clarification_node(state):
    return {"messages": [AIMessage(content="I need more details. What tech stack or seniority level are you looking for?")]}

# NEW: This node now acts as the "Resume" point after human intervention
def human_review_node(state):
    # If we are here, it means the human has updated the state!
    # We essentially "approved" the candidate or provided a manual override.
    return {"messages": [AIMessage(content="✅ Human verified. Proceeding with this candidate.")]}

def final_answer_node(state):
    candidate = state["selected_candidate"]
    text = f"✅ Best Match: {candidate.get('best_candidate_name')}\n"
    text += f"📊 Score: {candidate.get('match_score')}/100\n"
    text += f"💡 Evidence: {candidate.get('evidence')}"
    return {"messages": [AIMessage(content=text)]}

# --- 4. BUILD THE GRAPH ---
workflow = StateGraph(RecruitmentState)

workflow.add_node("classify_intent", classify_intent_node)
workflow.add_node("answer_follow_up", answer_follow_up_node)
workflow.add_node("analyze_query", analyze_query)
workflow.add_node("ask_clarification", ask_clarification_node)
workflow.add_node("retrieve_resumes", retrieve_resumes)
workflow.add_node("grade_candidates", grade_candidates)
workflow.add_node("human_review", human_review_node)
workflow.add_node("final_answer", final_answer_node)

workflow.set_entry_point("classify_intent")

workflow.add_conditional_edges("classify_intent", route_by_intent, 
    {"FOLLOW_UP": "answer_follow_up", "NEW_SEARCH": "analyze_query"})

workflow.add_conditional_edges("analyze_query", decide_next_step, 
    {"ask_clarification": "ask_clarification", "retrieve_resumes": "retrieve_resumes"})

workflow.add_edge("retrieve_resumes", "grade_candidates")

workflow.add_conditional_edges("grade_candidates", check_confidence, 
    {"human_review": "human_review", "final_answer": "final_answer"})

workflow.add_edge("answer_follow_up", END)
workflow.add_edge("ask_clarification", END)
workflow.add_edge("final_answer", END)
workflow.add_edge("human_review", END) # After review, we end (or could loop back)

# --- 5. COMPILE WITH INTERRUPT ---
checkpointer = MemorySaver()

# 🔥 CRITICAL CHANGE: We interrupt BEFORE 'human_review' runs.
app = workflow.compile(
    checkpointer=checkpointer, 
    interrupt_before=["human_review"] 
)

# --- 6. RUN APP (INTERACTIVE LOOP) ---
if __name__ == "__main__":
    print("🤖 HITL Recruitment Agent Ready. (Type 'quit' to exit)")
    config = {"configurable": {"thread_id": "session_1"}}

    while True:
        try:
            # 1. Check if we are currently paused (Waiting for Human)
            current_state = app.get_state(config)
            
            # If the next step is 'human_review', we are PAUSED.
            if current_state.next and current_state.next[0] == "human_review":
                print("\n⚠️  LOW CONFIDENCE DETECTED.")
                print("The agent is unsure. It wants to proceed to 'human_review'.")
                user_action = input("Type 'ok' to approve the candidate, or 'override' to reject: ")
                
                if user_action.lower() == "ok":
                    # We resume execution (NULL input acts as "continue")
                    print("👍 Approving...")
                    # Update confidence manually so we don't loop forever if we changed logic
                    app.update_state(config, {"confidence": "HIGH"}) 
                    
                    # Resume graph!
                    for event in app.stream(None, config=config):
                        for key, value in event.items():
                             print(f"   ↳ {key}...")
                             if "messages" in value: print(f"🤖 Agent: {value['messages'][-1].content}")
                             
                else:
                    print("❌ Rejected. Cancelling this search.")
                    # We can just break or reset here. For now, let's just wait for new input.
                    pass
                
                continue # Skip the normal input loop

            # 2. Normal Operation (Not Paused)
            user_input = input("\nHR User: ")
            if user_input.lower() in ["quit", "exit"]:
                break
            
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