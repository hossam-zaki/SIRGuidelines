import streamlit as st
from llama_index.core import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
import openai

openai.api_key = st.secrets.openai_key
if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Give me a patient presentation and I'll tell you the guidelines"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the guidelines"):
        reader = SimpleDirectoryReader(input_dir="./guidelines", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()

chat_engine = index.as_chat_engine(chat_mode="context", 
                                   verbose=True,
                                   system_prompt=("You are an expert in surgical guidelines. You will be given patient presentations. For each presentation, generate 3 sections for each guideline recommendation, Medical, Laboratory, and Antibiotics. For Medical Recommendations, include exact information about how long to withhold and reinitiate medication. For Laboratory recommendations, include exact values for INR and PLT. For Antibiotic recommendations, include information about whether Routine Prophylaxis is recommended, and if so, which one."))

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history