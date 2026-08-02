import time
import logging
from typing import Optional
from groq import Groq
from pipeline.llm.config import LLMConfig
from pipeline.llm.models import LLMResponse
from pipeline.prompt.models import PromptPackage

logger = logging.getLogger(__name__)

class BaseProvider:
    def generate(self, prompt_package: PromptPackage) -> LLMResponse:
        raise NotImplementedError

class GroqProvider(BaseProvider):
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = None
        if self.config.api_key:
            self.client = Groq(api_key=self.config.api_key, max_retries=self.config.retry_count)
        else:
            logger.warning("Groq API key is missing!")

    def generate(self, prompt_package: PromptPackage) -> LLMResponse:
        start_time = time.time()
        
        if not self.client:
            return LLMResponse(
                provider="groq",
                model=self.config.model_name,
                response="",
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                latency_ms=0,
                finish_reason="error",
                success=False,
                error="API key is missing or client failed to initialize."
            )
            
        messages = [
            {"role": "system", "content": prompt_package.system_prompt},
            {"role": "user", "content": f"Context Information:\n{prompt_package.context}\n\nQuestion:\n{prompt_package.user_question}"}
        ]
        
        try:
            completion = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                timeout=self.config.timeout,
            )
            
            latency = int((time.time() - start_time) * 1000)
            
            usage = completion.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
            total_tokens = usage.total_tokens if usage else 0
            
            return LLMResponse(
                provider="groq",
                model=self.config.model_name,
                response=completion.choices[0].message.content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                latency_ms=latency,
                finish_reason=completion.choices[0].finish_reason,
                success=True
            )
            
        except Exception as e:
            latency = int((time.time() - start_time) * 1000)
            logger.error(f"Groq API Error: {str(e)}")
            return LLMResponse(
                provider="groq",
                model=self.config.model_name,
                response="",
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                latency_ms=latency,
                finish_reason="error",
                success=False,
                error=str(e)
            )
