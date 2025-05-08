import streamlit as st
import os
import dotenv
import uuid
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from rag_methods import stream_llm_response, load_predefined_docs_and_urls, stream_llm_rag_response

dotenv.load_dotenv()

# Set USER_AGENT for WebBaseLoader
os.environ["USER_AGENT"] = "InsuranceRAGChatbot/1.0"

MODELS = [
    "openai/gpt-3.5-turbo",
]

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon=":robot:",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.html("""<h2 style="text-align:center;"><i>Insurance and Investment RAG Chatbot</i></h2>""")

# API Key Input
st.subheader("Enter Your OpenAI API Key")
openai_api_key = st.text_input(
    "OpenAI API Key (required)",
    type="password",
    placeholder="Enter your key starting with 'sk-'",
    help="Get your API key from https://platform.openai.com/api-keys"
)

# Validate API Key
missing_openai = openai_api_key == "" or openai_api_key is None or not openai_api_key.startswith("sk-")

if missing_openai:
    st.warning("Please enter a valid OpenAI API key to use the chatbot.")
else:
    # Setup
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Hello! I'm a chatbot for insurance and investment queries. Ask me a question based on the provided documents and URLs, and I'll answer if the information is available."}
        ]

    # Load predefined files and URLs with API key
    if "vector_db" not in st.session_state:
        load_predefined_docs_and_urls(openai_api_key)

    # Sidebar
    with st.sidebar:
        st.divider()
        models = []
        for model in MODELS:
            models.append(model)

        st.selectbox(
            "🤖 Select a Model",
            options=models,
            key="model",
        )

        cols0 = st.columns(2)
        with cols0[0]:
            is_vector_db_loaded = ("vector_db" in st.session_state and st.session_state.vector_db is not None)
            st.toggle(
                "Use RAG",
                value=is_vector_db_loaded,
                key="use_rag",
                disabled=not is_vector_db_loaded,
            )

        with cols0[1]:
            st.button("Clear Chat", on_click=lambda: st.session_state.messages.clear(), type="primary")

        st.header("RAG Sources:")
        st.write("Using predefined insurance and investment documents and URLs.")

    # Main chat app
    model_provider = st.session_state.model.split("/")[0]
    if model_provider == "openai":
        llm_stream = ChatOpenAI(
            api_key=openai_api_key,
            model_name=st.session_state.model.split("/")[-1],
            temperature=0.3,
            streaming=True,
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Your message"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            messages = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages]
            if not st.session_state.use_rag:
                st.write_stream(stream_llm_response(llm_stream, messages))
            else:
                st.write_stream(stream_llm_rag_response(llm_stream, messages))