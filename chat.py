import ollama
import streamlit as st 
import yaml

with open("situation.yaml","r") as file:
    situation = yaml.safe_load(file)
    location = situation["location"]
    filepath = situation['filepath']
    avator_profile = situation["avator_profile"]
    first_question = situation["first_question"]


st.title("Your AI Girlfriend")
print(filepath)

header_image = filepath
st.image(header_image,caption='Situation',use_column_width=True)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role":"system","content":avator_profile}]
    st.session_state["messages"] = [{"role":"assistant","content":first_question}]

### write Message History
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message(msg["role"],avatar="👨‍💻").write(msg["content"])
    else:
        st.chat_message(msg["role"],avatar="😀").write(msg["content"])

## Generator for Streaming Tokens

def generate_response():

    response = ollama.chat(model='llama3', stream=True,messages=st.session_state.messages)
    print(response)
    for partial_resp in response:
        token = partial_resp["message"]["content"]
        st.session_state["full_message"] += token
        yield token
    
    if prompt := st.chat_input():
        st.session_state.messages.append({"role":"user","content":prompt})
        st.chat_message("user",avatar="👨‍💻").write(prompt)
        st.session_state["full_message"] = ""
        st.chat_message("assistant",avatar="😀").write_stream(generate_response)
        st.session_state.messages.append({"role":"assistant","content":st.session_state["full_message"]})
        








































import ollama
import streamlit as st
import yaml

with open("situation.yaml", "r") as file:
    situation = yaml.safe_load(file)
    location = situation["location"]
    filepath = situation["filepath"]
    avator_profile = situation["avator_profile"]
    first_question = situation["first_question"]

#--------------------------------------------

st.title("Your AI GirlFriend")
print(filepath)
# Add a header image
header_image = filepath
st.image(header_image, caption='Situation', use_column_width=True)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "system", "content": avator_profile}]
    st.session_state["messages"] = [{"role": "assistant", "content": first_question}]

### Write Message History
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message(msg["role"], avatar="🧑‍💻").write(msg["content"])
    else:
        st.chat_message(msg["role"], avatar="😃").write(msg["content"])


## Generator for Streaming Tokens
def generate_response():
    
    response = ollama.chat(model='llama3', stream=True, messages=st.session_state.messages)
    print(response)
    for partial_resp in response:
        token = partial_resp["message"]["content"]
        st.session_state["full_message"] += token
        yield token

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="🧑‍💻").write(prompt)
    st.session_state["full_message"] = ""
    st.chat_message("assistant", avatar="😃").write_stream(generate_response)
    st.session_state.messages.append({"role": "assistant", "content": st.session_state["full_message"]})
