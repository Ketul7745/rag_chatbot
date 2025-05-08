import os
import dotenv
from time import time
import streamlit as st
import glob
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

dotenv.load_dotenv()

os.environ["USER_AGENT"] = "InsuranceRAGChatbot/1.0"

def stream_llm_response(llm_stream, messages):
    response_message = ""
    for chunk in llm_stream.stream(messages):
        response_message += chunk.content
        yield chunk 
    st.session_state.messages.append({"role": "assistant", "content": response_message})

def load_predefined_docs_and_urls(openai_api_key):
    docs = []
    
    # Predefined file paths (replace with your actual file paths)
    file_paths = [
        "./docs/test_rag.docx",  # Example DOCX file
        "./docs/test_rag.pdf",   # Example PDF file
    ]
    
    # Predefined URLs (replace with your actual URLs)
    urls = [
        "https://www.insurancejournal.com/news/national/2023/09/15/739874.htm",
        "https://www.investopedia.com/articles/investing/110613/top-5-mutual-fund-holders-aig.asp"
    ]
    
    # Load files
    for file_path in file_paths:
        try:
            if not os.path.exists(file_path):
                st.warning(f"File not found: {file_path}")
                continue
            if file_path.endswith(".txt"):
                loader = TextLoader(file_path)
            elif file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            else:
                st.warning(f"Unsupported file type: {file_path}")
                continue
            docs.extend(loader.load())
        except Exception as e:
            st.error(f"Error loading file {file_path}: {e}")
            print(f"Error loading file {file_path}: {e}")
    
    # Load URLs
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            docs.extend(loader.load())
        except Exception as e:
            st.error(f"Error loading URL {url}: {e}")
            print(f"Error loading URL {url}: {e}")
    
    if docs:
        _split_and_load_docs(docs, openai_api_key)
        st.toast("Predefined files and URLs loaded successfully.", icon="✅")
    else:
        st.error("No documents or URLs loaded successfully.")

def initialize_vector_db(docs, openai_api_key):
    # Create FAISS index
    vector_db = FAISS.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(api_key=openai_api_key)
    )
    # Save FAISS index to disk
    index_name = f"faiss_index_{str(time()).replace('.', '')[:14]}_{st.session_state['session_id']}"
    vector_db.save_local(f"./faiss_db/{index_name}")
    
    # Manage FAISS indexes (limit to 20)
    index_files = glob.glob("./faiss_db/faiss_index_*")
    index_names = sorted([os.path.basename(f) for f in index_files if os.path.isdir(f)])
    print("Number of FAISS indexes:", len(index_names))
    while len(index_names) > 20:
        old_index = index_names.pop(0)
        import shutil
        shutil.rmtree(f"./faiss_db/{old_index}")
    
    return vector_db

def _split_and_load_docs(docs, openai_api_key):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=1000,
    )
    document_chunks = text_splitter.split_documents(docs)
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = initialize_vector_db(document_chunks, openai_api_key)
    else:
        # FAISS doesn't support direct add_documents; create new index
        existing_docs = st.session_state.vector_db.index.reconstruct_n(0, st.session_state.vector_db.index.ntotal)
        new_vector_db = FAISS.from_documents(
            documents=document_chunks,
            embedding=OpenAIEmbeddings(api_key=openai_api_key)
        )
        st.session_state.vector_db = new_vector_db

def _get_context_retriever_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name='messages'),
        ("user", "{input}"),
        ("user", "Given the above context, generate a search query to look up in order to get relevant information")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_rag_chain(llm):
    retriever_chain = _get_context_retriever_chain(st.session_state.vector_db, llm)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
        """You are a helpful assistant, you will have to answer the 
        user's questions strictly based on the context provided. If the 
        answer to the question is not in the context, say you dont know.
        {context} """),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def stream_llm_rag_response(llm_stream, messages):
    conversation_rag_chain = get_conversational_rag_chain(llm_stream)
    response_message = "*(RAG Response)*\n"
    for chunk in conversation_rag_chain.pick("answer").stream({"messages": messages[:-1], "input": messages[-1].content}):
        response_message += chunk
        yield chunk
    st.session_state.messages.append({"role": "assistant", "content": response_message})