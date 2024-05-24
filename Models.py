from langchain_community.llms import HuggingFaceEndpoint
import os
import json
# Now we can override it and set it to "AI Assistant"
from langchain_core.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st



# with open('apikey.json') as f:
#     data = json.load(f)

HUGGINGFACEHUB_API_TOKEN = st.secrets['hugging-api']
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
    
def get_key():
    HUGGINGFACEHUB_API_TOKEN = st.secrets['hugging-api']
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
     
    return HUGGINGFACEHUB_API_TOKEN

def call_gemini():    
    llm = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=st.secrets['gemini-api'])

    return llm

def call_llama3():
    
    print('Llama')

    llm = HuggingFaceEndpoint(
                    endpoint_url=f"https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct",
                    max_new_tokens=128,
                    max_length=128,
                    # top_k=10,
                    # top_p=0.95,
                    # typical_p=0.95,
                    temperature=0.01,
                    # repetition_penalty=1.03,

                    )
    return llm

def call_phi_mini_4k():

    print("PHI-mini3")
    llm = HuggingFaceEndpoint(
                    endpoint_url=f"https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct",
                    max_new_tokens=128,
                    max_length=128,
                    # top_k=10,
                    # top_p=0.95,
                    # typical_p=0.95,
                    temperature=0.01,
                    # repetition_penalty=1.03,

                    )
    
    return llm
