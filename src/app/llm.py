import os

from dotenv import load_dotenv

load_dotenv()

if os.getenv("LLM_MODE") == "fake":
    # CI: a scripted, deterministic model. No API key, no network. The grader drives the REAL agent loop and tools with it.
    from app.fake_llm import ScriptedChatModel

    llm = fast_llm = agent_llm = ScriptedChatModel()
else:
    from langchain_groq import ChatGroq

    llm = ChatGroq(model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"), temperature=0)
    # Small, cheap model for simple single-shot work (profile merge): shows the per-model cost split in Grafana
    fast_llm = ChatGroq(model=os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b"), temperature=0)
    # The agent answers briefly: capping output keeps each request inside small per-request token limits (Groq free tier)
    agent_llm = ChatGroq(model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"), temperature=0, max_tokens=int(os.getenv("AGENT_MAX_TOKENS", "1500")))
