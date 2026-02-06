"""
OpenAI-compatible LLM client for remote inference via vLLM or similar APIs.

Provides:
- LLMResponse dataclass for structured responses
- LLMClient with retry logic and health checking
"""

import time
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI, APIError, APIConnectionError, RateLimitError, APITimeoutError

from .exceptions import GenerationError
from .logging_config import get_logger

logger = get_logger("llm_client")


@dataclass
class LLMResponse:
    """Structured response from LLM generation."""
    text: str
    input_tokens: int
    output_tokens: int
    generation_time: float
    model: str


class LLMClient:
    """
    OpenAI-compatible LLM client with retry logic.

    Usage:
        client = LLMClient(base_url, api_key, model)
        response = client.generate("What is FIBO?")
        print(response.text)
    """

    TRANSIENT_ERRORS = (APIConnectionError, RateLimitError, APITimeoutError)

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_retries: int = 3,
        timeout: float = 120.0,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout

        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )

    def generate(
        self,
        question: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            question: The question to answer
            context: Optional context to include
            system_prompt: Optional system prompt override
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = deterministic)

        Returns:
            LLMResponse with text and usage stats

        Raises:
            GenerationError: If generation fails after all retries
        """
        if system_prompt is None:
            if context:
                system_prompt = (
                    "You are a financial expert. "
                    "Answer based ONLY on the provided context. Be concise."
                )
            else:
                system_prompt = "You are a financial expert. Answer concisely."

        messages = [{"role": "system", "content": system_prompt}]

        if context:
            messages.append({
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            })
        else:
            messages.append({"role": "user", "content": question})

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                start_time = time.time()

                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                generation_time = time.time() - start_time

                choice = response.choices[0]
                usage = response.usage

                return LLMResponse(
                    text=choice.message.content.strip() if choice.message.content else "",
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                    generation_time=generation_time,
                    model=response.model or self.model,
                )

            except self.TRANSIENT_ERRORS as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = min(2.0 * (2 ** (attempt - 1)), 30.0)
                    logger.warning(
                        f"API call attempt {attempt} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)

            except APIError as e:
                raise GenerationError(
                    f"API error: {e}",
                    model_id=self.model,
                    max_tokens=max_tokens,
                    original_error=e,
                )

        raise GenerationError(
            f"Generation failed after {self.max_retries} attempts",
            model_id=self.model,
            max_tokens=max_tokens,
            original_error=last_error,
        )

    def health_check(self) -> bool:
        """
        Check if the API endpoint is reachable.

        Returns:
            True if the API responds successfully
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
                temperature=0.0,
            )
            return True
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
