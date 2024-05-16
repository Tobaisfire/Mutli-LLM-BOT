import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from time import sleep

# Define the Streamlit sidebar and its elements
with st.sidebar:
    st.markdown("*Tobis* is **really** ***cool*** 🤩.")
    st.markdown('''
        :red[Tobis] :orange[is] :green[your] :blue[virtual] :violet[assistant]
        :gray[created by] :rainbow[Keval] 😎 ''')
    st.markdown("Here's a bouquet For You Guys &mdash;\
                :tulip::cherry_blossom::rose::hibiscus::sunflower::blossom:")
    
    st.subheader('Models and parameters')
    selected_model = st.sidebar.selectbox('Choose a Large Language model', ['Gemini by Google', 'Llama3 8B Parameter by Meta'], key='selected_model')

    st.write('This chatbot is created using the open-source Llama 2 LLM model from Meta and Gemini pro 1 by google.')
    
    st.success('API key already provided!', icon='✅')

    st.success(f"Selected Model : {selected_model}")

# Initialize the conversation components
template = """The following is a friendly conversation between a human and an AI name Tobis created by Keval Saud. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{chat_history}
Human: {input}
AI Assistant:"""

memory = ConversationBufferMemory(memory_key="chat_history")

PROMPT = PromptTemplate(input_variables=["chat_history", "input"], template=template)

# Define the conversation chain based on the selected model
if selected_model == 'Gemini by Google':
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=st.secrets['gemini-api'])
elif selected_model == 'Llama3 8B Parameter by Meta':
    HUGGINGFACEHUB_API_TOKEN = st.secrets['hugging-api']
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
    llm = HuggingFaceEndpoint(
                endpoint_url=f"https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct",
                max_new_tokens=512,
                max_length=128,
                temperature=0.01,
                repetition_penalty=1.03,
                )
else:
    st.error("Invalid model selection!")

conversation = ConversationChain(
    prompt=PROMPT,
    llm=llm,
    verbose=False,
    memory=memory,
)

# Define the Streamlit app
st.title("💬 Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello 😎 Let's start Talking !!. I am Tobis "}]

# Display chat messages and handle user input
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    with st.spinner("Typing....."):
        response = conversation.predict(input=prompt)
        response = response.split("\n\n")[0]
    
    def stream_data():
        for word in response.split(" "):
            yield word + " "
            sleep(0.05)

    st.session_state["messages"].append({"role": "assistant", "content": response})       
    st.chat_message("assistant").write_stream(stream_data)
