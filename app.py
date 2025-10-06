import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferMemory
from utils.tools import load_tools
from utils.explain import explain_sql, explain_results

llm = ChatOpenAI(model="gpt-4o-mini")
tools = load_tools()
memory = ConversationBufferMemory(memory_key="chat_history")

agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",  # multi-step reasoning
    memory=memory,
    verbose=True
)

st.title("Agentic Data Analyst")

query = st.text_input("Ask a question about the database or knowledge base")

if query:
    response = agent.run(query)

    if isinstance(response, dict):
        sql = response.get("query", "")
        result = response.get("result", "")
        st.write("### Answer", result)
        st.code(sql, language="sql")
        st.write("**SQL Explanation:**", explain_sql(sql))
        st.write("**Interpretation:**", explain_results(sql, result))
    else:
        st.write("### Answer", response)
