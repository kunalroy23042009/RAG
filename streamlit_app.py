"""
Chat UI — ask questions about your processed notices, ChatGPT-style.

Entry point for the whole Streamlit app. Run from the project root:
    streamlit run streamlit_app.py

The Admin Portal page (pages/1_Admin_Portal.py) is auto-discovered by
Streamlit and shown in the sidebar — no extra setup needed.
"""
import streamlit as st

from src.ui_common import get_retriever
from src.retrieval import generate_augmented_response_stream

st.set_page_config(page_title="Academic Notice Assistant", page_icon="💬", layout="centered")

st.title("💬 Academic Notice Assistant")
st.caption("Ask questions about every notice that's been processed so far.")

if "messages" not in st.session_state:
    st.session_state.messages = []

retriever = get_retriever()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_prompt = st.chat_input("Ask about a notice, e.g. 'which exam is on 20.09.2025'")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        full_response = st.write_stream(
            generate_augmented_response_stream(user_prompt, retriever)
        )

    st.session_state.messages.append({"role": "assistant", "content": full_response})

with st.sidebar:
    st.header("Academic Notice RAG")
    st.write(f"Messages this session: {len(st.session_state.messages)}")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption("Need to add new notices? Use the **Admin Portal** page in the sidebar above.")