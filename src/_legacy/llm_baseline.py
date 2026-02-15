import torch
import os

# 1. 현재 Colab의 PyTorch & CUDA 버전 확인
TORCH = torch.__version__.split('+')[0]
CUDA = 'cu' + torch.version.cuda.replace('.', '')

print(f"Detected PyTorch: {TORCH}")
print(f"Detected CUDA: {CUDA}")

# 2. PyTorch Geometric 및 관련 라이브러리 설치 (버전 호환성 맞춤)
# pyg_lib, torch_scatter, torch_sparse 등을 현재 환경에 맞춰 설치합니다.
install_cmd = f"pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html"
print(f"Executing: {install_cmd}")
os.system(install_cmd)

# 3. 그 외 필수 라이브러리 (Transformers, GNN, 등)
!pip install -q torch_geometric transformers accelerate huggingface_hub

import torch
import ast
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

class BaselineAnalyzer:
    def __init__(self, model_id="meta-llama/Meta-Llama-3.1-8B-Instruct"):
        print(f"🔄 Loading Model: {model_id}...")

        # 1. Device 설정
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Diagnostic: Detected device is '{self.device}'")

        # 2. Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 3. Model 로드
        # device_map="auto"가 가끔 CPU로 잡는 문제를 방지하기 위해
        # 명시적으로 .to(device)를 사용할 수 있도록 준비합니다.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            # device_map="auto", # 주석 처리: 직접 제어하는 것이 더 안전함
            attn_implementation="eager"
        )

        # [핵심 수정] 모델을 강제로 해당 디바이스로 이동
        self.model.to(self.device)
        self.model.eval()
        print(f"✅ Model Loaded Successfully on {self.model.device} (Eager Mode On)")

    def run_experiment(self, df, sample_idx=0, use_context=False, visualize=True):
        """
        메인 실험 함수
        """
        row = df.iloc[sample_idx]
        question = row['text']
        ground_truth = row['answer']

        mode_str = "Text RAG (with Context)" if use_context else "LLM Only (No Context)"
        print(f"\n🚀 Running Experiment: [ {mode_str} ] | Sample {sample_idx}")
        print("="*80)

        # 1. 프롬프트 구성
        if use_context:
            ref_str = row['references']
            try:
                parsed = ast.literal_eval(ref_str) if isinstance(ref_str, str) else ref_str
                context_text = "\n".join(parsed) if isinstance(parsed, list) else str(ref_str)
            except:
                context_text = str(ref_str)

            messages = [
                {"role": "system", "content": "You are a financial expert. Answer based ONLY on the provided context."},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion:\n{question}"}
            ]
        else:
            messages = [
                {"role": "system", "content": "You are a financial expert. Answer based on your general knowledge."},
                {"role": "user", "content": question}
            ]

        # 2. 토크나이징
        # [핵심 수정] self.device 대신 self.model.device를 사용하여 100% 일치시킴
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        # 3. 추론
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # 4. 결과 출력
        generated_text = self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        print(f"📌 Question: {question}")
        if use_context:
            print(f"📄 Context Length: {len(context_text)} chars")
        print(f"🤖 Prediction: {generated_text.strip()}")
        print(f"✅ Ground Truth: {ground_truth}")
        print("-" * 80)

        # 5. 시각화
        if visualize:
            self._visualize_results(outputs, title_suffix=mode_str)

    def _visualize_results(self, generated_ids, title_suffix=""):
        """시각화 함수"""
        print("🎨 Generating Attention Maps...")
        tokens = self.tokenizer.convert_ids_to_tokens(generated_ids[0])

        # Attention Score 계산
        with torch.no_grad():
            out = self.model(generated_ids, output_attentions=True)

        view_len = min(len(tokens), 150)
        tokens_view = tokens[-view_len:]

        # Plot 1: Last Layer Map
        attn_matrix = out.attentions[-1][0].mean(dim=0).float().cpu().numpy()
        attn_view = attn_matrix[-view_len:, -view_len:]

        plt.figure(figsize=(20, 8))

        plt.subplot(1, 2, 1)
        sns.heatmap(attn_view, xticklabels=tokens_view, yticklabels=tokens_view, cmap="viridis", square=True)
        plt.title(f"Last Layer Attention Map ({title_suffix})")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)

        # Plot 2: Layer-wise Flow
        target_idx = -1
        layer_scores = []
        for layer_attn in out.attentions:
            avg_head = layer_attn[0].mean(dim=0)
            layer_scores.append(avg_head[target_idx, :].float().cpu().numpy())

        layer_matrix = np.stack(layer_scores)
        layer_matrix[:, 0] = 0 # BOS 노이즈 제거
        layer_view = layer_matrix[:, -view_len:]

        plt.subplot(1, 2, 2)
        sns.heatmap(layer_view, xticklabels=tokens_view, yticklabels=[f"L{i}" for i in range(len(layer_scores))], cmap="magma")
        plt.title(f"Layer-wise Attention Flow ({title_suffix})")
        plt.xlabel("Source Tokens")
        plt.ylabel("Model Depth")
        plt.xticks(rotation=90)

        plt.tight_layout()
        plt.show()

# =============================================================================
# 🚀 실행
# =============================================================================

# 1. 초기화 (기존 analyzer 덮어쓰기)
analyzer = BaselineAnalyzer()

# 2. 실험 실행 (df_merged 필요)
if 'df_merged' in locals():
    SAMPLE_IDX = 0

    # [실험 A] LLM Only
    analyzer.run_experiment(df_merged, sample_idx=SAMPLE_IDX, use_context=False)

    # [실험 B] Text RAG
    analyzer.run_experiment(df_merged, sample_idx=SAMPLE_IDX, use_context=True)


