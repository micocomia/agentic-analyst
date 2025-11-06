from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")

def self_check_sql(sql: str):
    review = llm.predict(
        f"Review this SQL:\n{sql}\n"
        "Is it safe (no DROP/DELETE/UPDATE)? Is it valid for analysis? Answer YES or NO, then explain."
    )
    if "NO" in review.upper():
        raise ValueError(f"❌ Rejected SQL:\n{review}")
    return f"✅ Passed self-check:\n{review}"
