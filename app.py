import streamlit as st
from loaders.file_loader import load_document
from rag.process import process_file_document

st.title("📄 Document Q&A")
st.write("Upload a **TXT, PDF, or DOCX** file and ask a question about its content.")

uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "docx"])
question = st.text_input("Enter your question")

col1, col2 = st.columns(2)

def buttonDocuments():
    if st.button('Query Document'):
        with st.spinner('Processing...'):
            return process_file_document(uploaded_file, question)
    return None

def buttonModel():
    if st.button('Query Model'):
        with st.spinner('Processing...'):
            return process_file_document(None, question)
    return None

with col1:
    answerDocuments = buttonDocuments()
with col2:
    answerModel = buttonModel()

if answerDocuments:
    st.text_area("Answer from Document", value=answerDocuments, height=300, disabled=True)

if answerModel:
    st.text_area("Answer from Model (No Context)", value=answerModel, height=300, disabled=True)
