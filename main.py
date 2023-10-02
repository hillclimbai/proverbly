import streamlit as st
import langchain_helper as helper
import textwrap

db = helper.create_vector_db_from_proverbs()

st.set_page_config(page_title="📚 Proverbly")
st.title("📚 Proverbly")
st.subheader("Get advice from the book of Proverbs")

with st.form(key="form"):
        query = st.text_area('What are you struggeling with:')
        submit_button = st.form_submit_button(label="Submit")

if query:
    response = helper.get_response_from_query(db, query)
    st.subheader("Answer")
    st.text(textwrap.fill(response))
