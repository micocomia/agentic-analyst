from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")

def explain_sql(sql: str):
    return llm.predict(f"Explain this SQL in plain English:\n{sql}")

def explain_results(sql: str, results: str):
    return llm.predict(
        f"SQL: {sql}\nResults: {results}\n"
        "Explain in business-friendly terms what these results mean."
    )
