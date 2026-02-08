"""
Attention score extraction and analysis for local HuggingFace models.

Extracts generation→context attention maps to understand which graph context
tokens the model attends to when generating answers.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .config import AttentionConfig
from .logging_config import get_logger

logger = get_logger("attention")


@dataclass
class AttentionResult:
    """Extracted attention scores for a single generation."""
    context_attention_scores: np.ndarray  # [num_context_tokens] aggregated
    context_tokens: List[str]
    layer_attention: Optional[Dict[int, np.ndarray]] = None  # layer_idx -> [num_ctx_tokens]
    entity_attention: Optional[Dict[str, float]] = None  # entity_name -> score
    top_k_tokens: Optional[List[Tuple[str, float]]] = None
    per_head_attention: Optional[Dict[int, Dict[int, np.ndarray]]] = None  # {layer: {head: [num_ctx_tokens]}}


class AttentionExtractor:
    """Extracts and processes attention scores from HuggingFace model outputs."""

    def __init__(self, config: AttentionConfig):
        self.config = config

    @torch.no_grad()
    def extract(
        self,
        model,
        input_ids: torch.Tensor,
        tokenizer,
        context_token_range: Tuple[int, int],
        generated_token_start: int,
        entity_names: Optional[List[str]] = None,
    ) -> AttentionResult:
        """
        Extract attention from generated tokens to context tokens.

        Args:
            model: HuggingFace model with output_attentions support
            input_ids: Full sequence (prompt + generated) [1, seq_len]
            tokenizer: Tokenizer for decoding tokens
            context_token_range: (start, end) indices of context in input_ids
            generated_token_start: Index where generated tokens begin
            entity_names: Optional entity names for aggregation

        Returns:
            AttentionResult with extracted scores
        """
        ctx_start, ctx_end = context_token_range
        num_ctx_tokens = ctx_end - ctx_start

        if num_ctx_tokens <= 0:
            logger.warning("No context tokens to extract attention from")
            return AttentionResult(
                context_attention_scores=np.array([]),
                context_tokens=[],
            )

        # Forward pass with attention
        outputs = model(input_ids, output_attentions=True)
        attentions = outputs.attentions  # tuple of [batch, heads, seq, seq]

        # Decode context tokens
        ctx_token_ids = input_ids[0, ctx_start:ctx_end]
        context_tokens = [tokenizer.decode([tid]) for tid in ctx_token_ids]

        # Extract specified layers
        layers_to_use = self.config.layers_to_extract
        num_layers = len(attentions)
        layer_indices = [idx % num_layers for idx in layers_to_use]

        layer_attention_map = {}
        all_layer_scores = []
        per_head_map: Dict[int, Dict[int, np.ndarray]] = {}
        per_head_mode = self.config.aggregate_heads == "none"

        gen_end = input_ids.shape[1]

        for layer_idx in layer_indices:
            attn = attentions[layer_idx]  # [1, heads, seq, seq]

            if per_head_mode:
                # Per-head: extract each head individually
                num_heads = attn.shape[1]
                head_scores_map: Dict[int, np.ndarray] = {}
                head_scores_list = []

                for h in range(num_heads):
                    if generated_token_start < gen_end:
                        gen_to_ctx = attn[0, h, generated_token_start:gen_end, ctx_start:ctx_end]
                        h_scores = gen_to_ctx.mean(dim=0).cpu().numpy()
                    else:
                        h_scores = np.zeros(num_ctx_tokens)
                    head_scores_map[h] = h_scores
                    head_scores_list.append(h_scores)

                per_head_map[layer_idx] = head_scores_map
                # Layer-level aggregate = mean across heads (for backward compat)
                ctx_scores = np.mean(head_scores_list, axis=0)
            else:
                # Aggregate heads as before
                if self.config.aggregate_heads == "mean":
                    attn_agg = attn[0].mean(dim=0)  # [seq, seq]
                else:  # max
                    attn_agg = attn[0].max(dim=0).values  # [seq, seq]

                # Slice: generated tokens -> context tokens
                if generated_token_start < gen_end:
                    gen_to_ctx = attn_agg[generated_token_start:gen_end, ctx_start:ctx_end]
                    ctx_scores = gen_to_ctx.mean(dim=0).cpu().numpy()
                else:
                    ctx_scores = np.zeros(num_ctx_tokens)

            layer_attention_map[layer_idx] = ctx_scores
            all_layer_scores.append(ctx_scores)

        # Free GPU memory from attentions
        del attentions, outputs
        torch.cuda.empty_cache()

        # Average across layers
        context_attention_scores = np.mean(all_layer_scores, axis=0)

        # Top-k tokens
        top_k = self.config.top_k_tokens
        if len(context_attention_scores) > 0:
            top_indices = np.argsort(context_attention_scores)[-top_k:][::-1]
            top_k_list = [
                (context_tokens[i], float(context_attention_scores[i]))
                for i in top_indices
                if i < len(context_tokens)
            ]
        else:
            top_k_list = []

        # Entity aggregation
        entity_attn = None
        if entity_names:
            entity_attn = self.aggregate_to_entities(
                context_attention_scores, context_tokens, entity_names
            )

        return AttentionResult(
            context_attention_scores=context_attention_scores,
            context_tokens=context_tokens,
            layer_attention=layer_attention_map if not self.config.save_context_attention_only else None,
            entity_attention=entity_attn,
            top_k_tokens=top_k_list,
            per_head_attention=per_head_map if per_head_mode else None,
        )

    @staticmethod
    def aggregate_to_entities(
        token_scores: np.ndarray,
        token_strings: List[str],
        entity_names: List[str],
    ) -> Dict[str, float]:
        """Aggregate token-level attention to entity-level scores."""
        entity_scores: Dict[str, float] = {}
        token_text_lower = [t.strip().lower() for t in token_strings]

        for entity in entity_names:
            entity_lower = entity.strip().lower()
            entity_tokens = entity_lower.split()
            score = 0.0
            count = 0

            for i, tok in enumerate(token_text_lower):
                for et in entity_tokens:
                    if et and et in tok:
                        score += float(token_scores[i])
                        count += 1
                        break

            entity_scores[entity] = score / max(count, 1)

        return entity_scores

    @staticmethod
    def save_attention(result: AttentionResult, path: str) -> None:
        """Save attention result to .npz file."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "context_attention_scores": result.context_attention_scores,
        }
        if result.layer_attention:
            for layer_idx, scores in result.layer_attention.items():
                save_dict[f"layer_{layer_idx}"] = scores

        if result.per_head_attention:
            for layer_idx, heads in result.per_head_attention.items():
                for head_idx, scores in heads.items():
                    save_dict[f"layer_{layer_idx}_head_{head_idx}"] = scores

        np.savez_compressed(str(save_path), **save_dict)

        # Save metadata as JSON alongside
        meta_path = save_path.with_suffix(".json")
        meta = {
            "context_tokens": result.context_tokens,
            "top_k_tokens": result.top_k_tokens,
            "entity_attention": result.entity_attention,
            "has_per_head": result.per_head_attention is not None,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.debug(f"Attention saved: {save_path}")

    @staticmethod
    def load_attention(path: str) -> AttentionResult:
        """Load attention result from .npz file."""
        data = np.load(path)
        scores = data["context_attention_scores"]

        layer_attention = {}
        per_head_attention: Dict[int, Dict[int, np.ndarray]] = {}

        for key in data.files:
            if key == "context_attention_scores":
                continue
            # Per-head keys: "layer_X_head_Y"
            if "_head_" in key:
                parts = key.split("_")  # ["layer", X, "head", Y]
                layer_idx = int(parts[1])
                head_idx = int(parts[3])
                if layer_idx not in per_head_attention:
                    per_head_attention[layer_idx] = {}
                per_head_attention[layer_idx][head_idx] = data[key]
            elif key.startswith("layer_"):
                layer_idx = int(key.split("_")[1])
                layer_attention[layer_idx] = data[key]

        # Load metadata
        meta_path = Path(path).with_suffix(".json")
        context_tokens = []
        top_k_tokens = None
        entity_attention = None
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            context_tokens = meta.get("context_tokens", [])
            top_k_tokens = meta.get("top_k_tokens")
            entity_attention = meta.get("entity_attention")

        return AttentionResult(
            context_attention_scores=scores,
            context_tokens=context_tokens,
            layer_attention=layer_attention if layer_attention else None,
            entity_attention=entity_attention,
            top_k_tokens=top_k_tokens,
            per_head_attention=per_head_attention if per_head_attention else None,
        )
