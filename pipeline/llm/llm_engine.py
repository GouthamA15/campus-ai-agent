import logging
from pipeline.prompt.models import PromptPackage
from pipeline.llm.models import LLMResponse
from pipeline.llm.config import LLMConfig
from pipeline.llm.providers import GroqProvider

logger = logging.getLogger(__name__)

class LLMEngine:
    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig()
        
        if self.config.provider == "groq":
            self.provider = GroqProvider(self.config)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
            
    def process(self, prompt_package: PromptPackage) -> LLMResponse:
        logger.info(f"Generating response using {self.config.provider} ({self.config.model_name})...")
        
        response = self.provider.generate(prompt_package)
        
        if response.success:
            logger.info(f"LLM Engine: SUCCESS (Latency: {response.latency_ms}ms, Input Tokens: {response.input_tokens}, Output Tokens: {response.output_tokens})")
        else:
            logger.error(f"LLM Engine: FAILED ({response.error})")
            
        return response
