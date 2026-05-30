import streamlit as str
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import load_prompt
from dotenv import load_dotenv

import warnings

# Silence deprecation warnings specifically from the transformers library
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

load_dotenv()

str.header("Research Paper Summarizer")

input_topic=str.selectbox("Choose The Topic",[
    "The Impact of Distributed Ledger Technology on Supply Chain Transparency",
    "Optimizing Transformer Models for Low-Resource Language Translation",
    "Microplastic Accumulation in Freshwater Ecosystems and Its Biomagnification",
    "The Role of Sleep Architecture in Enhancing Neuroplasticity and Memory Consolidation",
    "Comparative Analysis of Solid-State Electrolytes for Next-Generation Lithium-Metal Batteries"
])

input_style=str.selectbox("Choose the style",[
    "Academic and scholarly (peer-reviewed tone, dense vocabulary, objective)",
    "Sarcastic and witty (humorous, ironic undertones, playful banter)",
    "Corporate and professional (business casual, concise, action-oriented)",
    "Poetic and metaphorical (rich imagery, rhythmic structure, expressive)",
    "Minimalist and direct (short sentences, zero fluff, high impact)",
    "Gothic horror (dark adjectives, suspenseful pacing, eerie atmosphere)",
    "ELI5 / Explaining to a child (simple analogies, zero jargon, conversational)",
    "Cyberpunk / Sci-fi (gritty, high-tech terminology, neon-soaked descriptions)"
])

input_length=str.selectbox("Choose The length context",["Single-sentence / Punchy (under 20 words)",
    "Short paragraph / Micro-blog (50 to 100 words)",
    "Standard essay / Medium-form (300 to 500 words)",
    "Long-form / Comprehensive guide (1,000 to 1,500 words)",
    "Ultra-short / Headline or tweet format (under 280 characters)"])


model=ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite',model_kwargs={'device':0})

prompt_val=load_prompt("prompt.json")


prompt=prompt_val.invoke({
    "topic":input_topic,
    "style":input_style,
    "length":input_length
})

if str.button("Summarize"):
    result=model.invoke(prompt)
    str.write(result.text)


