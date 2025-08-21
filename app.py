import streamlit as st
import pandas as pd
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline


# APP SETUP
st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("🤖 RAG Chatbot By Peter Inaolaji")
st.write("This application combines Retrieval-Augmented Generation (RAG) with the power of large language models to give you accurate, context-aware answers based on your documents.")

# Helper function to build RAG
def build_rag_pipeline(df: pd.DataFrame):
    df.columns = [col.strip() for col in df.columns]

    # Convert rows to natural language documents
    def row_to_document(row):
        return ", ".join([f"{col}: {row[col]}" for col in df.columns])

    documents = [Document(page_content=row_to_document(row)) for _, row in df.iterrows()]

    # Split into chunks
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(documents)

    # Embeddings + Vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    # Load LLM
    qa_model = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)
    llm = HuggingFacePipeline(pipeline=qa_model)

    # Create RAG chain
    retriever = vectorstore.as_retriever(search_type="similarity", k=3)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )

    return qa_chain

# File Upload Section
uploaded_file = st.file_uploader("📂 Upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data loaded successfully!")
    st.write("Here are the first 5 rows of your data:")
    st.dataframe(df.head())

    # Cache RAG pipeline per uploaded dataset
    @st.cache_resource
    def get_pipeline(dataframe):
        return build_rag_pipeline(dataframe)

    qa_chain = get_pipeline(df)

  
    # Chatbox Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    if query := st.chat_input("Ask me anything about your dataset..."):
        # Display user message
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        # Get RAG answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = qa_chain.invoke(query)
                answer = result["result"] if isinstance(result, dict) else result
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info(" Please upload a CSV file to begin.")
