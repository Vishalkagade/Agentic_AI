from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver


load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

class JokeState(TypedDict):
    joke: str
    topic: str
    explanation: str

def create_joke(state: JokeState):
    prompt = f"create a joke using {state['topic']}"
    response = llm.invoke(prompt).content

    return {"joke":response}

def create_explanation(state: JokeState):
    prompt = f"Please explain me the {state['joke']}"
    response = llm.invoke(prompt).content

    return {"explanation":response}


graph = StateGraph(JokeState)

graph.add_node("create_joke", create_joke)
graph.add_node("create_explanation", create_explanation)

graph.add_edge(START, "create_joke")
graph.add_edge("create_joke", "create_explanation")
graph.add_edge("create_explanation", END)

checkpointer = MemorySaver()

workflow = graph.compile(checkpointer)


initial_state = {"topic" : "Pune"}

config = {"configurable": {"thread_id": "1"}}

final_state = workflow.invoke(initial_state, config=config)

print(final_state["joke"])
print("----------------")
print(list(workflow.get_state_history(config)))