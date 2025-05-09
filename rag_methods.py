import os
import dotenv
from time import time
import streamlit as st
import glob
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from time import sleep

dotenv.load_dotenv()

os.environ["USER_AGENT"] = "InsuranceRAGChatbot/1.0"

def is_allowed(url, user_agent=os.getenv("USER_AGENT", "InsuranceRAGChatbot/1.0")):
    """Check if URL is allowed by robots.txt."""
    parsed_url = urlparse(url)
    robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
    rp = RobotFileParser()
    try:
        response = requests.get(robots_url, headers={"User-Agent": user_agent}, timeout=5)
        rp.parse(response.text.splitlines())
        return rp.can_fetch(user_agent, url)
    except Exception:
        return True  # Allow if robots.txt is inaccessible

def get_urls_from_page(url, base_domain, support_path, visited, max_urls=500):
    """Extract URLs from a single page, staying within /support."""
    if len(visited) >= max_urls:
        return set()
    
    urls = set()
    try:
        response = requests.get(url, headers={"User-Agent": os.getenv("USER_AGENT", "InsuranceRAGChatbot/1.0")}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        for link in soup.find_all("a", href=True):
            href = link["href"]
            absolute_url = urljoin(url, href)
            parsed_url = urlparse(absolute_url)
            
            # Ensure URL is from the same domain, under /support, and not a fragment
            if (parsed_url.netloc == base_domain and
                absolute_url.startswith(f"https://www.angelone.in{support_path}") and
                absolute_url not in visited and
                not parsed_url.fragment):
                if is_allowed(absolute_url):
                    urls.add(absolute_url)
    except Exception as e:
        print(f"Error crawling {url}: {e}")
    
    return urls

def crawl_support_urls(seed_url="https://www.angelone.in/support", support_path="/support", max_urls=20, delay=1.0):
    """Crawl all URLs under Angel One's support section."""
    parsed_seed = urlparse(seed_url)
    base_domain = parsed_seed.netloc
    visited = set()
    to_visit = {seed_url}
    all_urls = set()
    
    while to_visit and len(all_urls) < max_urls:
        current_url = to_visit.pop()
        if current_url in visited:
            continue
            
        print(f"Crawling: {current_url}")
        visited.add(current_url)
        new_urls = get_urls_from_page(current_url, base_domain, support_path, visited, max_urls)
        all_urls.add(current_url)
        to_visit.update(new_urls - visited)
        
        # time.sleep(delay)  # Respectful crawling
    
    return sorted(list(all_urls))

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
        "./docs/America's_Choice_2500_Gold_SOB (1) (1).pdf",
        "./docs/America's_Choice_5000_Bronze_SOB (2).pdf",
        "./docs/America's_Choice_5000_HSA_SOB (2).pdf",
        "./docs/America's_Choice_7350_Copper_SOB (1) (1).pdf",
        "./docs/America's_Choice_Medical_Questions_-_Modified_(3) (1).docx"
    ]
    
    # Predefined URLs
    predefined_urls = [
        # "https://www.insurancejournal.com/news/national/2023/09/15/739874.htm",
        # "https://www.investopedia.com/articles/investing/110613/top-5-mutual-fund-holders-aig.asp"
    ]
    
    # Crawl URLs from Angel One support
    crawled_urls = crawl_support_urls()
    
    # Combine predefined and crawled URLs
    urls = list(set(predefined_urls + crawled_urls))
    
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
    vector_db = FAISS.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(api_key=openai_api_key)
    )
    index_name = f"faiss_index_{str(time()).replace('.', '')[:14]}_{st.session_state['session_id']}"
    vector_db.save_local(f"./faiss_db/{index_name}")
    
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