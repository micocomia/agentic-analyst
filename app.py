import os
import re
from dotenv import load_dotenv

# Load environment variables safely
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "utils", ".env"))

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from utils.tools import load_tools
#from utils.explain import explain_sql, explain_results

# Initialize components
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = load_tools()
memory = ConversationBufferMemory(memory_key="chat_history")

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# --- Streamlit UI ---
st.title("Agentic Data Analyst")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if "sql" in msg:
                st.code(msg["sql"], language="sql")
            if "sql_explanation" in msg:
                st.write("**SQL Explanation:**", msg["sql_explanation"])
            if "results_explanation" in msg:
                st.write("**Interpretation:**", msg["results_explanation"])

# --- Chat input ---
query = st.chat_input("Ask a question about the knowledge base")

if query:
    # Show user query
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Run the agent
    response = agent.run(query)
    assistant_msg = {"role": "assistant"}

    # Case 1: agent returned dict (custom tools)
    if isinstance(response, dict):
        sql = response.get("query", "")
        result = response.get("result", "")
        assistant_msg["content"] = f"### Answer\n{result}"
        assistant_msg["sql"] = sql
        #assistant_msg["sql_explanation"] = explain_sql(sql)
        #assistant_msg["results_explanation"] = explain_results(sql)
    else:
        # Case 2: plain text or SQL in markdown
        sql_pattern = re.compile(r"```sql\s*(.*?)```", re.IGNORECASE | re.DOTALL)
        match = sql_pattern.search(response)

        if match:
            sql = match.group(1).strip()
            remaining_text = sql_pattern.sub("", response).strip()
            assistant_msg["content"] = f"### Answer\n{remaining_text}"
            assistant_msg["sql"] = sql
            #assistant_msg["sql_explanation"] = explain_sql(sql)
            #assistant_msg["results_explanation"] = explain_results(sql)
        else:
            assistant_msg["content"] = f"### Answer\n{response}"

    # Display assistant message
    with st.chat_message("assistant"):
        st.markdown(assistant_msg["content"])
        if "sql" in assistant_msg:
            st.code(assistant_msg["sql"], language="sql")
            #st.write("**SQL Explanation:**", assistant_msg["sql_explanation"])
            #st.write("**Interpretation:**", assistant_msg["results_explanation"])

    # Store message
    st.session_state["messages"].append(assistant_msg)
