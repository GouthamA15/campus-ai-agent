import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class LLMConfig:
    provider: str = os.getenv("LLM_PROVIDER", "groq")
    model_name: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    api_key: str = os.getenv("GROQ_API_KEY", "")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    top_p: float = float(os.getenv("LLM_TOP_P", "0.9"))
    timeout: int = int(os.getenv("LLM_TIMEOUT", "30"))
    retry_count: int = int(os.getenv("LLM_RETRY_COUNT", "2"))
