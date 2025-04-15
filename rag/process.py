import os
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from loaders.file_loader import load_document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def process_file_document(uploaded_file, question):
    model_local = Ollama(model="tinyllama")
    retriever = ""

    if uploaded_file:
        docs = load_document(uploaded_file)
        if docs is None:
            return "Unsupported file type."

        print("🔹 Splitting document...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
        doc_splits = text_splitter.split_documents(docs)
        print(f"✅ Total Chunks: {len(doc_splits)}")

        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # Check if vectorstore already exists
        if os.path.exists("vectorstore/index/index.faiss"):
            print("Loading saved FAISS vectorstore...")
            vectorstore = FAISS.load_local("vectorstore/index", embeddings=embedding, allow_dangerous_deserialization=True)
        else:
            print("Creating FAISS vectorstore from documents...")
            vectorstore = FAISS.from_documents(doc_splits, embedding=embedding)
            vectorstore.save_local("vectorstore/index")

        retriever = vectorstore.as_retriever()

    if retriever == "":
        template = """Answer the question:
        Question: {question}
        """
        context_question = {"question": RunnablePassthrough()}
    else:
        template = """Use the below context to answer the user's question. Be accurate and concise.
        {context}
        Question: {question}
        """
        context_question = {"context": retriever, "question": RunnablePassthrough()}

    after_rag_prompt = ChatPromptTemplate.from_template(template)
    chain = context_question | after_rag_prompt | model_local | StrOutputParser()

    print("🤖 Invoking model...")
    response = chain.invoke(question)
    return chain.invoke(response)
