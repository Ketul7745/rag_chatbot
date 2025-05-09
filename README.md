# rag_chatbot

Hello everyone, this is a financial and insuarance chatbot using rag and deployed using streamlit.

Hosted Link: https://ragchatbot-3ybqqg7vbsjsvkyc65irt5.streamlit.app/

Requirements:
  1. OpenAI API Key

How to run this:
  1. Clik the hosted link above
  2. Enter your open ai api key (your account should have access to gpt-3.5-turbo)
  3. wait for 60-90 seconds
  4. a chat interface will be displayed and now you can ask your questions

The chatbot uses specific data in the "docs" folder and also uses website urls present in crawled_url.txt file.

Technologies and libraries used: 
  1. Langchain
  2. Faiss for vectordb
  3. beautifulsoup for scraping of urls
  4. streamlit for deployment of webapp

Limitations and possible future scope:
  1. Currently the project only supports 50 urls as of now due to token limits
  2. project was tested only on gpt-3.5-turbo which poses token limitations
  3. better error handling
  4. using advanced rag for better context window
  5. dynamic crawling of urls to refresh the urls when changed
