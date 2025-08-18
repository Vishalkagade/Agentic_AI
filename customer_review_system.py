from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from dotenv import load_dotenv

from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field
import operator

from langchain_openai import ChatOpenAI
from IPython.display import Image,display

load_dotenv()

model = ChatOpenAI(model_name="gpt-4o-mini")

class sentimentSchema(BaseModel):
    sentiment : Literal["Positive", "Negative"]


class DiagnosisSchema(BaseModel):
    issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"] = Field(description='The category of issue mentioned in the review')
    tone: Literal["angry", "frustrated", "disappointed", "calm"] = Field(description='The emotional tone expressed by the user')
    urgency: Literal["low", "medium", "high"] = Field(description='How urgent or critical the issue appears to be')


structured_model = model.with_structured_output(sentimentSchema)
structured_model2 = model.with_structured_output(DiagnosisSchema)


class customerReviewState(TypedDict):
    review: str
    sentiment : Literal["Positive", "Negative"]
    diagnosis : dict
    responce : str

def sentiment_analysis(state: customerReviewState) -> dict:
    prompt = f"please find the sentiment of the following review - {state["review"]}"
    sentiment = structured_model.invoke(prompt).sentiment

    return {"sentiment":sentiment}

def check_sentiment(state: customerReviewState) ->  Literal["Positive_responce", "run_diagnosis"]:

    if state["sentiment"] == "Positive_responce":
        return "Positive_responce"
    
    else:
        return "run_diagnosis"

def Positive_responce(state: customerReviewState):
    prompt = f"Write a thank you msg to use for the {state["review"]} and ask them to give use a feedback!"
    responce = structured_model.invoke(prompt)
    return {"responce":responce}

def run_diagnosis(state: customerReviewState):
    prompt = f"""Diagnose this negative review:\n\n{state['review']}\n"
    "Return issue_type, tone, and urgency."""
    response = structured_model2.invoke(prompt)

    return {"diagnosis": response.model_dump()}

def negative_responce(state: customerReviewState):
    diagnosis = state['diagnosis']

    prompt = f"""You are a support assistant.
    The user had a '{diagnosis['issue_type']}' issue, sounded '{diagnosis['tone']}', and marked urgency as '{diagnosis['urgency']}'.
    Write an empathetic, helpful resolution message."""
        
    responce = model.invoke(prompt).content

    return {"responce":responce}



graph = StateGraph(customerReviewState)
graph.add_node("sentiment_analysis", sentiment_analysis)
graph.add_node("Positive_responce",Positive_responce)
graph.add_node("run_diagnosis",run_diagnosis)
graph.add_node("negative_responce",negative_responce)

graph.add_edge(START, "sentiment_analysis")
graph.add_conditional_edges("sentiment_analysis",check_sentiment)

graph.add_edge("Positive_responce",END)
graph.add_edge("run_diagnosis","negative_responce")

graph.add_edge("negative_responce", END)

workflow = graph.compile()

with open("graph.png", "wb") as f:
    f.write(workflow.get_graph().draw_mermaid_png())
print("Graph saved as graph.png")

intial_state={
    'review': "I’ve been trying to log in for over an hour now, and the app keeps freezing on the authentication screen. I even tried reinstalling it, but no luck. This kind of bug is unacceptable, especially when it affects basic functionality."
}
final_state = workflow.invoke(intial_state)
print(final_state["responce"])



# prompt = {"review" : "The software is too bad"}

# output = workflow.invoke(prompt)

# print(output)