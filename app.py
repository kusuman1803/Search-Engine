import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.schema import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# Setup Tools
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

search = DuckDuckGoSearchRun(name="Search")

st.set_page_config(page_title="LangChain Chat Search", page_icon="🔎")
st.title("🔎 LangChain - Chat With Search")

# Sidebar for settings
st.sidebar.title("⚙️ Settings")
api_key = st.sidebar.text_input("Enter your GROQ API key:", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot that can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

def get_chat_history():
    history = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        else:
            history.append(AIMessage(content=msg["content"]))
    return history

if prompt := st.chat_input(placeholder="Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
        streaming=True
    )

    tools = [search, arxiv, wiki]

    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True
    )

    # Generate agent response
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(
            input=prompt,
            callbacks=[st_cb],
            chat_history=get_chat_history()
        )
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
