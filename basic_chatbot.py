from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from typing import TypedDict
from dotenv import load_dotenv

from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field
import operator

from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI()

# create a state

class chatBotState(TypedDict):
    message : Annotated[list[BaseMessage], add_messages]

# create a chat-node 

def chat_node(state: chatBotState):

    message = state["message"]

    responce = model.invoke(message)

    return {"message": responce}

checkpointer = MemorySaver()

graph = StateGraph(chatBotState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node",END)

chatBot = graph.compile(checkpointer=checkpointer)

thread_id = "1"

while True:
    user_input = input("Type your quary: ")
    print("user_input", user_input)

    if user_input.strip().lower() in ["exit", "bye", "end"]:
        break

    config = {"configurable": {"thread_id": thread_id}}
    responce = chatBot.invoke({"message":[SystemMessage(content="I am Vishal's assistant"),HumanMessage(content=user_input)]},config=config)

    print("AI:", responce["message"][-1].content)

