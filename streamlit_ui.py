"""Use this file to generate the Streamlit UI"""

import streamlit as st
content = ""
with open("README.md", "r") as f:
    content = f.read()

st.markdown(content)
