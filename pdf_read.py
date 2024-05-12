from pypdf import PdfReader as pdf
import sys
import streamlit as st

st.title("PDF Reader")
file = st.file_uploader("Upload a PDF", type="pdf")

def read_txt_pdf(file):
    text = pdf(file)
    doc_len = len(text.pages)
    doc = ""
    text_on_page = ""
    for page_no in range(len(text.pages)):
        page = text.pages[page_no]
        text_on_page = page.extract_text()
        doc += text_on_page
    return doc
    #print(doc)


if file:
    text = read_txt_pdf(file)
    st.text_area("Text", text, height=400)