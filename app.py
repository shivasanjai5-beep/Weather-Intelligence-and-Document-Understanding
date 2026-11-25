"""Streamlit UI. Run with:  streamlit run app.py"""
import streamlit as st

import document_agent
import weather_agent
from router import route
from web_search import web_answer

st.set_page_config(page_title="Weather & Document Chatbot", page_icon="🤖")
st.title("🤖 Weather & Document Chatbot")

# ---- sidebar: document upload ----
with st.sidebar:
    st.header("Upload document")
    file = st.file_uploader("PDF or TXT (e.g. your resume)", type=["pdf", "txt"])
    if file and st.session_state.get("doc_name") != file.name:
        st.session_state.store = document_agent.DocumentStore(document_agent.read_file(file))
        st.session_state.doc_name = file.name
    if "store" in st.session_state:
        st.success(f"Loaded: {st.session_state.doc_name}")

if "messages" not in st.session_state:
    st.session_state.messages = []


def show(msg):
    """Draw one chat message with its tables, charts and sources."""
    st.write(msg["text"])
    for t in msg.get("tables", []):
        st.dataframe(t, use_container_width=True)
    for f in msg.get("figs", []):
        st.plotly_chart(f, use_container_width=True)
    if msg.get("sources"):
        with st.expander("Sources"):
            for s in msg["sources"]:
                st.write(s)


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        show(m)

question = st.chat_input("Ask about weather, your document, or anything else...")
if question:
    if len(question.strip()) < 3:
        st.warning("Please type a longer question.")
        st.stop()

    st.session_state.messages.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            decision = route(question, "store" in st.session_state)
            tool = decision["tool"]
            st.caption(f"🔧 Tool chosen by LLM: **{tool}**")

            if tool == "weather":
                reply = weather_agent.run(question, decision.get("city", ""), decision.get("when", ""))
            elif tool == "document":
                if "store" in st.session_state:
                    reply = document_agent.answer(question, st.session_state.store)
                else:
                    reply = {"text": "Please upload a document first (sidebar)."}
            else:
                reply = web_answer(question)
                reply["text"] = "🌐 *External information (web):*\n\n" + reply["text"]

        reply["role"] = "assistant"
        show(reply)
    st.session_state.messages.append(reply)
