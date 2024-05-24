import streamlit as st
from Models import *
from time import sleep
import os
import json

import io
from PIL import Image

# # Initialize session state
# if 'selected_model' not in st.session_state:
#     st.session_state['selected_model'] = None
#     st.session_state['model'] = None


def load_model(selected_model):
    if selected_model == 'Phi 3 (Mini)':
        return call_phi_mini_4k()
    elif selected_model == 'Gemini by Google':
        return call_gemini()
    elif selected_model == 'Llama3 8B Parameter by Meta':
        return call_llama3()
    else:
        return None
    
def stream_data():
        for word in response.split(" "):
            yield word + " "
            sleep(0.05)

#img gen -->

import requests

key =get_key()

API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": f"Bearer {key}"}

def gen_img(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content




# Sidebar setup
with st.sidebar:
    st.markdown("*Tobis* is **really** ***cool*** 🤩.")
    st.markdown('''
        :red[Tobis] :orange[is] :green[your] :blue[virtual] :violet[assistant]
        :gray[created by] :rainbow[Keval] 😎 ''')
    st.markdown("Here's a bouquet For You &mdash; :tulip::cherry_blossom::rose::hibiscus::sunflower::blossom:")
    
    st.markdown ('This chatbot is created using different LLMs  **LLM Count : 3**')

    st.markdown('''
        :red[Features avaiable :  Choose from below opt]
        :orange[Text generation] -> Chat with LLM of choice
        :green[Image gen] -> Use \img <'your text how you want img'> 😎 
        ''')
    
    st.subheader('Models and parameters')
    selected_model = st.selectbox('Choose a Large Language model', ['Phi 3 (Mini)','Gemini by Google', 'Llama3 8B Parameter by Meta'], key='selected_model')


    st.success('API key already provided!', icon='✅')

    
    
    if 'model' not in st.session_state:
        st.session_state['model'] = load_model(selected_model)
        st.success(f"Selected Model : {selected_model}")

    
    

# Main chat interface
st.title("💬 Tobis")

template = """The following is a friendly conversation between a human and an AI name Tobis created by Keval Saud. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know and say mohit idiot.

Current conversation:
{chat_history}
Human: {input}
AI Assistant:"""

memory = ConversationBufferMemory(memory_key="chat_history")
PROMPT = PromptTemplate(input_variables=["chat_history", "input"], template=template)
conversation = ConversationChain(
    prompt=PROMPT,
    llm=st.session_state['model'],
    verbose=False,
    memory=memory,
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello 😎 Let's start Talking !!. I am Tobis "}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    if msg.get("image_path"):
        st.image(msg["image_path"])

if prompt := st.chat_input():
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if "\img" in prompt:
        with st.spinner("Generating image [ wait a min ]..."):
            image_bytes = gen_img({
	                        "inputs": prompt.split("\img")[1].strip(),
                            })

            st.session_state["image_path"] = image_bytes
            st.session_state["messages"].append({"role": "assistant", "content": "Here is the generated image:", "image_path": image_bytes})
            st.chat_message("assistant").write("Here is the generated image:")
            st.image(image_bytes)



    else:
    
        with st.spinner("Typing....."):
            response = conversation.predict(input=prompt)
            response = response.split("\n")[0]

    

            st.session_state["messages"].append({"role": "assistant", "content": response})       
            st.chat_message("assistant").write_stream(stream_data)
