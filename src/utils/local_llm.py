"""
Local HuggingFace model manager for multi-model comparison experiments.

Supports loading/unloading models with quantization, generating with attention
extraction, and managing GPU memory across model switches.
"""

import gc
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from .attention import AttentionConfig, AttentionExtractor, AttentionResult
from .config import ModelConfig
from .logging_config import get_logger

logger = get_logger("local_llm")


MODEL_REGISTRY: Dict[str, dict] = {
    "llama8b": {
        "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "type": "dense",
        "default_quant": None,
    },
    "llama70b": {
        "model_id": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "type": "dense",
        "default_quant": "4bit",
    },
    "mixtral": {
        "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "type": "moe",
        "default_quant": "4bit",
    },
    "qwen_moe": {
        "model_id": "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        "type": "moe",
        "default_quant": None,
    },
}


@dataclass
class LocalLLMResponse:
    """Structured response from local HuggingFace model generation."""
    text: str
    input_tokens: int
    output_tokens: int
    generation_time: float
    model: str
    attention_data: Optional[AttentionResult] = None
    context_token_range: Optional[Tuple[int, int]] = None


class LocalLLMManager:
    """Manages local HuggingFace model loading, generation, and memory."""

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.model = None
        self.tokenizer = None
        self.current_model_id: Optional[str] = None
        self.current_alias: Optional[str] = None
        self.attention_extractor: Optional[AttentionExtractor] = None

    def load_model(
        self,
        model_alias: str,
        quant_override: Optional[str] = None,
        attention_config: Optional[AttentionConfig] = None,
    ) -> None:
        """
        Load a model from the registry.

        Args:
            model_alias: Key in MODEL_REGISTRY (e.g. "llama8b")
            quant_override: Override quantization ("4bit", "8bit", None for bf16)
            attention_config: Config for attention extraction
        """
        if model_alias not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model alias '{model_alias}'. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )

        # Unload current model if different
        if self.current_alias == model_alias and self.model is not None:
            logger.info(f"Model '{model_alias}' already loaded, skipping")
            return
        if self.model is not None:
            self.unload_model()

        registry_entry = MODEL_REGISTRY[model_alias]
        model_id = registry_entry["model_id"]
        quant = quant_override or registry_entry.get("default_quant")

        logger.info(f"Loading model: {model_alias} ({model_id}), quant={quant}")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        # HF token for gated models
        hf_token = os.environ.get("HF_TOKEN")
        token_kwargs = {"token": hf_token} if hf_token else {}

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **token_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Build quantization config
        load_kwargs = {
            "device_map": self.model_config.llm_device_map,
            "attn_implementation": "eager",  # Required for output_attentions
            **token_kwargs,
        }

        if quant == "4bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif quant == "8bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        self.model.eval()

        self.current_model_id = model_id
        self.current_alias = model_alias

        # Setup attention extractor
        if attention_config and attention_config.enabled:
            self.attention_extractor = AttentionExtractor(attention_config)

        param_count = sum(p.numel() for p in self.model.parameters())
        if torch.cuda.is_available():
            mem_gb = torch.cuda.memory_allocated() / 1e9
            logger.info(
                f"Model loaded: {model_alias} | {param_count:,} params | "
                f"GPU memory: {mem_gb:.1f} GB"
            )
        else:
            logger.info(f"Model loaded: {model_alias} | {param_count:,} params (CPU)")

    def unload_model(self) -> None:
        """Unload current model and free GPU memory."""
        if self.model is not None:
            alias = self.current_alias or "unknown"
            logger.info(f"Unloading model: {alias}")
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.current_model_id = None
        self.current_alias = None
        self.attention_extractor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.info("Model unloaded, GPU memory freed")

    def generate(
        self,
        question: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        extract_attention: bool = False,
        attention_config: Optional[AttentionConfig] = None,
        entity_names: Optional[List[str]] = None,
    ) -> LocalLLMResponse:
        """
        Generate a response with optional attention extraction.

        Args:
            question: The question to answer
            context: Optional graph/text context
            system_prompt: Optional system prompt
            few_shot_examples: List of {"role": ..., "content": ...} message pairs
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            extract_attention: Whether to extract attention scores
            attention_config: Attention extraction config
            entity_names: Entity names for attention aggregation

        Returns:
            LocalLLMResponse with text, usage, and optional attention
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        # Build messages
        messages = self._build_messages(question, context, system_prompt, few_shot_examples)

        # Tokenize
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        # Find context token range if context provided
        context_token_range = None
        if context and extract_attention:
            context_token_range = self._find_context_range(prompt_text, context, inputs["input_ids"])

        # Generate
        start_time = time.time()
        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature

            output_ids = self.model.generate(inputs["input_ids"], **gen_kwargs)

        generation_time = time.time() - start_time

        # Decode response
        generated_ids = output_ids[0, input_len:]
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        output_tokens = len(generated_ids)

        # Extract attention if requested
        attention_data = None
        if extract_attention and context_token_range and self.attention_extractor:
            attn_config = attention_config or (
                self.attention_extractor.config if self.attention_extractor else None
            )
            if attn_config:
                try:
                    attention_data = self.attention_extractor.extract(
                        model=self.model,
                        input_ids=output_ids,
                        tokenizer=self.tokenizer,
                        context_token_range=context_token_range,
                        generated_token_start=input_len,
                        entity_names=entity_names,
                    )
                except Exception as e:
                    logger.warning(f"Attention extraction failed: {e}")

        # Clean up
        del inputs, output_ids
        torch.cuda.empty_cache()

        return LocalLLMResponse(
            text=response_text,
            input_tokens=input_len,
            output_tokens=output_tokens,
            generation_time=generation_time,
            model=self.current_alias or self.current_model_id or "unknown",
            attention_data=attention_data,
            context_token_range=context_token_range,
        )

    def _build_messages(
        self,
        question: str,
        context: Optional[str],
        system_prompt: Optional[str],
        few_shot_examples: Optional[List[Dict[str, str]]],
    ) -> List[Dict[str, str]]:
        """Build chat messages list."""
        if system_prompt is None:
            if context:
                system_prompt = (
                    "You are a financial expert. "
                    "Answer based ONLY on the provided context. Be concise."
                )
            else:
                system_prompt = "You are a financial expert. Answer concisely."

        messages = [{"role": "system", "content": system_prompt}]

        # Add few-shot examples
        if few_shot_examples:
            messages.extend(few_shot_examples)

        # Add the actual question
        if context:
            messages.append({
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            })
        else:
            messages.append({"role": "user", "content": question})

        return messages

    def _find_context_range(
        self,
        prompt_text: str,
        context: str,
        input_ids: torch.Tensor,
    ) -> Optional[Tuple[int, int]]:
        """Find token range of context within the prompt."""
        # Find context substring in prompt
        ctx_start_char = prompt_text.find(context)
        if ctx_start_char == -1:
            # Try truncated match
            ctx_start_char = prompt_text.find(context[:100])
            if ctx_start_char == -1:
                logger.debug("Could not locate context in prompt for attention extraction")
                return None

        # Encode prefix to find token offset
        prefix = prompt_text[:ctx_start_char]
        prefix_tokens = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
        ctx_tokens = self.tokenizer(context, add_special_tokens=False)["input_ids"]

        start_idx = len(prefix_tokens)
        end_idx = start_idx + len(ctx_tokens)

        # Clamp to input length
        seq_len = input_ids.shape[1]
        end_idx = min(end_idx, seq_len)

        if start_idx >= end_idx:
            return None

        return (start_idx, end_idx)
