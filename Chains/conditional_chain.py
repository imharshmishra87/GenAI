from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel,Field, field_validator
from typing import Literal, Annotated

load_dotenv()

model=GoogleGenerativeAI(model='gemini-3.1-flash-lite')

class Validate(BaseModel):
    sentiment : Annotated[str, Literal["Positive","Negative"], Field(description="Classify the sentiment as positive and negative")]

    @field_validator("sentiment")
    @classmethod

    def validation(cls , val):
        return val.title()


parser=StrOutputParser()

parser2=PydanticOutputParser(pydantic_object=Validate)


template1=PromptTemplate(
    template="Classify the following feedback into sentiment the sentiment will be positive or negative {feedback}\n {format_instructions}",
    input_variables=["feedback"],
    partial_variables={"format_instructions":parser2.get_format_instructions()}
)

classifier_chain=template1 | model | parser2

prompt2=PromptTemplate(
    template="Write an appropriate response for the following positive feedback -> {feedback}",
    input_variables=["feedback"]
)

prompt3=PromptTemplate(
    template="Write an appropriate response for the following negative feedback -> {feedback}",
    input_variables=["feedback"]
)

branch=RunnableBranch(
    (lambda x: x.sentiment=="Positive", prompt2 | model | parser),
    (lambda x: x.sentiment=="Negative", prompt3 | model | parser),
    RunnableLambda(lambda x : "Could not find the desired sentiment")
)

final_chain=classifier_chain | branch

feedback="""he Apple MacBook Pro M4 is a great workstation laptop. It can handle nearly every type of workload, including demanding tasks like video editing, 3D graphics, AI development, and scientific simulations. You can also do all of your color-sensitive work on the built-in display, as it has full coverage of the DCI-P3 color space and superb factory calibration. There's a wide port selection for peripherals and external monitors, including three USB-C/Thunderbolt 5s and an HDMI 2.1. Thermal throttling is minimal under load, and the laptop doesn't get overly hot or loud. Unfortunately, none of the components are user-replaceable, so you need to get the configuration that best suits your needs upfront."""

result=final_chain.invoke({"feedback":feedback})
print(result)

print(final_chain.get_graph().draw_ascii())