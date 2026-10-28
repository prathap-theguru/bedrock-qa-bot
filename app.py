"""A Streamlit interface for the documentation chatbot."""

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from chatbot import answer_question
from url_ingestion import ingest_url


st.set_page_config(page_title="Bedrock Docs Assistant", page_icon="💬")
load_dotenv(Path(__file__).with_name(".env"))

st.title("Bedrock Docs Assistant")
st.caption("Ask questions about your documentation or add another page.")

chat_tab, add_url_tab = st.tabs(["Chat", "Add URL"])

with chat_tab:
    question = st.chat_input("Ask a question about your documentation")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching documentation and preparing an answer…"):
                    answer = answer_question(question)
                st.markdown(answer)
            except Exception:
                st.error("Unable to generate an answer. Check your API access and configuration, then try again.")

with add_url_tab:
    st.subheader("Add a documentation page")
    st.caption("Public HTML pages only. Creating embeddings uses API credits.")
    with st.form("add_url"):
        url = st.text_input("Page URL", placeholder="https://example.com/documentation")
        submitted = st.form_submit_button("Create embeddings")
    if submitted:
        try:
            with st.spinner("Reading the page and storing embeddings…"):
                result = ingest_url(url.strip())
            if result.unchanged:
                st.success(f"This page is already stored and unchanged ({result.chunks} chunks).")
            else:
                st.success(f"Stored {result.chunks} chunks. You can now ask questions about this page.")
        except ValueError as error:
            st.error(str(error))
        except Exception:
            st.error("Unable to add the page. Check that the URL is public and your API accounts are available, then try again.")
