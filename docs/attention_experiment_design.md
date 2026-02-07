# LPG vs RDF Context가 LLM Attention에 미치는 영향 검증 실험

> **목표**: 그래프 표현 방식(LPG vs RDF)이 LLM 프롬프트로 직렬화될 때, attention 분포에 어떤 구조적 차이를 만드는지 정량적으로 검증한다.
>
> **선행 분석**: Neo4j DB 비교 + 토큰 수준 사고실험 기반
>
> **관련 문서**: [experiment_design.md](./experiment_design.md) (메인 실험 설계서)

---

## 1. 배경: 사고실험 요약

### 1.1 문제 제기

Graph RAG에서 그래프 데이터는 텍스트로 직렬화되어 LLM 프롬프트에 주입된다. 동일한 사실(fact)이라도 LPG 표현과 RDF 표현은 **토큰 수**, **의미 밀도**, **구조적 패턴**이 완전히 다르다. 이 차이가 LLM의 self-attention mechanism에 어떤 영향을 미치는가?

### 1.2 토큰 수준 관찰

동일한 7개 사실을 BPE 토크나이저로 분해한 결과:

```
LPG (113 tokens):  - HAS_CUSTOMER -> Apple (Company)
RDF (156 tokens):  ex:OurCompany fibo-fnd-rel-rel:hasCustomer ex:Apple
```

| 지표 | LPG | RDF |
|------|-----|-----|
| 총 토큰 (7 facts) | 113 | 156 |
| 1 fact당 토큰 | ~12 | ~19 |
| RDF/LPG 비율 | 1.00x | **1.38x** |
| FIBO prefix 비율 | 0% | **53%** (10/19 tokens) |

### 1.3 Attention에 대한 이론적 예측

**Softmax 경쟁**: attention weight = softmax(qK^T / sqrt(d_k)). 토큰이 많을수록 각 토큰의 attention weight가 희석된다. RDF의 prefix 토큰이 softmax 분모를 부풀려 핵심 토큰의 attention을 ~37% (12/19) 희석시킬 것으로 예측.

**Multi-head 전문화**: LPG에서는 entity type이 노드 옆에 괄호로 존재 → type 추적 head가 1-hop attention으로 작동. RDF에서는 type 정보가 별도 `rdf:type` 트리플로 분리 → type 추적 head 무력화. FIBO prefix 반복 패턴 인식에 일부 head가 낭비될 것으로 예측.

**Causal attention 흐름**: LPG의 계층적 구조(들여쓰기 + 앵커)는 positional encoding이 "소속" 관계를 암시. RDF의 평탄한 구조는 self-contained 트리플이므로 장거리 의존 불필요하나 주어 반복으로 토큰 낭비.

---

## 2. 가설

| ID | 가설 | 검증 메트릭 |
|----|------|------------|
| **H1** | 같은 context window에서 LPG가 ~38% 더 많은 사실을 담을 수 있다 | Semantic Density |
| **H2** | LPG context에서 answer-relevant 토큰의 attention weight가 더 높다 | Attention Entropy, Entity Coverage |
| **H3** | LPG는 zero/few-shot에 유리하고, RDF는 fine-tuning 후 격차가 줄어든다 | F1 by shot condition |
| **H4** | RDF의 self-contained 트리플이 hallucination을 줄일 수 있다 | EM, F1, ROUGE |
| **H5** | RDF context에서 prefix 인식에 낭비되는 attention head가 존재한다 | Per-Head Entropy, Prefix Waste Ratio |

---

## 3. 실험 설계

### 3.1 Independent Variable: Context Format (5 조건)

| 조건 | 코드명 | 설명 | 예시 |
|------|--------|------|------|
| **C1** | `lpg_structured` | LPG 데이터 → structured format | `[Company] Apple` / `NXP --[HAS_CUSTOMER]--> Apple` |
| **C2** | `lpg_natural` | LPG 데이터 → natural language format | `NXP has customer Apple.` |
| **C3** | `rdf_raw` | RDF 트리플 그대로 (FIBO URI 포함) | `(ex:Company, fibo-fnd-rel-rel:hasCustomer, ex:Apple)` |
| **C4** | `rdf_cleaned` | RDF prefix 제거 + camelCase 분리 | `(Company, has customer, Apple)` |
| **C5** | `no_context` | Context 없이 질문만 (baseline) | Question만 |

> **핵심 비교 쌍**:
> - C1 vs C3: LPG와 RDF의 "있는 그대로" 비교
> - C3 vs C4: prefix 제거만으로 attention이 개선되는지
> - C4 vs C1: prefix를 제거한 RDF가 LPG와 동등해지는지
> - C1 vs C2: 동일 LPG 데이터의 포맷 효과

### 3.2 Dependent Variables

#### Attention Metrics (새로 구현)

| 메트릭 | 수식 | 의미 | 검증 가설 |
|--------|------|------|-----------|
| **Attention Entropy** | H = -Σ p_i log₂(p_i) | 낮을수록 attention 집중 | H2, H5 |
| **Entity Coverage@K** | \|top-K tokens ∩ GT entities\| / \|GT entities\| | 정답 엔티티에 attend하는 비율 | H2 |
| **Prefix Waste Ratio** | Σ attn(prefix tokens) / Σ attn(all tokens) | FIBO prefix에 낭비되는 attention | H5 |
| **Semantic Density** | \|semantic tokens\| / \|total tokens\| | 의미 있는 토큰 비율 | H1 |
| **Per-Head Entropy Std** | std(H_head_1, ..., H_head_32) | head 전문화 정도 (낮으면 균일, 높으면 전문화) | H5 |

#### Answer Quality Metrics (기존 Evaluator 활용)

| 메트릭 | 도구 | 검증 가설 |
|--------|------|-----------|
| Exact Match | `Evaluator.compute_exact_match()` | H4 |
| Token F1 | `Evaluator.compute_token_f1()` | H3, H4 |
| ROUGE-L | `Evaluator.compute_rouge()` | H4 |

### 3.3 Control Variables

| 파라미터 | 값 | 이유 |
|----------|-----|------|
| Model | `llama8b` (primary) | 프로젝트 기본 모델 |
| Temperature | 0.0 | Deterministic generation |
| max_new_tokens | 256 | 충분한 답변 길이 |
| Attention layers | 전체 32 layers | Layer별 패턴 분석 |
| Attention heads | 전체 32 heads (개별 추출) | Head 전문화 분석 |
| Samples | 100개 (양쪽 DB 공통 질문) | 통계적 유의성 확보 |

### 3.4 샘플 선별 기준

양쪽 데이터베이스에 모두 서브그래프가 존재하는 질문만 사용:

```
finderlpg: Entity 노드에 question_ids 배열로 연결
finderrdf: Edge에 question_id 속성으로 연결

→ 교집합 질문 중 100개 무작위 샘플 (카테고리 균등 배분)
```

---

## 4. 구현 계획

### 4.1 기존 코드 활용

| 컴포넌트 | 파일 | 활용 방식 |
|----------|------|-----------|
| `AttentionExtractor` | `src/utils/attention.py` | Per-head 추출 확장 |
| `AttentionConfig` | `src/utils/config.py` | `aggregate_heads="none"` 옵션 추가 |
| `AttentionResult` | `src/utils/attention.py` | `per_head_attention` 필드 추가 |
| `LocalLLMManager` | `src/utils/local_llm.py` | `generate(extract_attention=True)` |
| `GraphFormatter` | `src/utils/formatting.py` | structured/natural/triple 포맷 |
| `Evaluator` | `src/utils/evaluation.py` | EM, F1, ROUGE 계산 |

### 4.2 새로 구현할 코드

#### (1) RDF Prefix Cleaner — `GraphFormatter`에 추가

```python
# src/utils/formatting.py — GraphFormatter 클래스에 추가

@classmethod
def clean_rdf_uri(cls, uri: str) -> str:
    """FIBO URI를 자연어로 변환.

    Examples:
        'fibo-fnd-rel-rel:hasCustomer' → 'has customer'
        'ex:Apple' → 'Apple'
        'fibo-fbc-fi-fi:hasOwnershipInterestIn' → 'has ownership interest in'
    """
    # prefix 제거: 마지막 ':' 뒤의 local name 추출
    if ":" in uri:
        local = uri.split(":")[-1]
    else:
        local = uri

    # camelCase → space separated lowercase
    import re
    result = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', local)
    return result.lower().strip()

@classmethod
def format_rdf_cleaned(cls, triples: List[Dict], max_triples: int = 50) -> str:
    """RDF 트리플을 prefix 제거 후 깔끔한 포맷으로 변환."""
    lines = ["Knowledge triples:"]
    for t in triples[:max_triples]:
        s = cls.clean_rdf_uri(t.get("source", "?"))
        p = cls.clean_rdf_uri(t.get("type", "related"))
        o = cls.clean_rdf_uri(t.get("target", "?"))
        lines.append(f"({s}, {p}, {o})")
    return "\n".join(lines)
```

#### (2) Per-Head Attention 추출 — `AttentionExtractor` 확장

```python
# src/utils/attention.py — AttentionExtractor.extract() 수정

# AttentionConfig에 추가:
#   aggregate_heads: str = "mean"  →  "mean" | "max" | "none"

# AttentionResult에 추가:
#   per_head_attention: Optional[Dict[int, Dict[int, np.ndarray]]] = None
#     → {layer_idx: {head_idx: [num_ctx_tokens]}}

# extract() 내부에서 aggregate_heads == "none"일 때:
if self.config.aggregate_heads == "none":
    head_map = {}
    for h in range(attn.shape[1]):  # num_heads
        gen_to_ctx = attn[0, h, gen_start:gen_end, ctx_start:ctx_end]
        head_map[h] = gen_to_ctx.mean(dim=0).cpu().numpy()
    layer_head_attention[layer_idx] = head_map
```

#### (3) AttentionAnalyzer — 새 파일

```python
# src/utils/attention_analysis.py

class AttentionAnalyzer:
    """Attention 분포 분석 메트릭 계산."""

    FIBO_PREFIXES = ["fibo", "fnd", "rel", "fbc", "fi", "ex:"]

    @staticmethod
    def entropy(scores: np.ndarray) -> float:
        """Attention distribution entropy (bits).
        낮을수록 attention이 집중됨.
        """
        p = scores / (scores.sum() + 1e-10)
        p = p[p > 1e-10]
        return float(-np.sum(p * np.log2(p)))

    @staticmethod
    def entity_coverage_at_k(
        token_scores: np.ndarray,
        token_strings: List[str],
        ground_truth_entities: List[str],
        k: int = 20,
    ) -> float:
        """Top-K attended 토큰이 GT 엔티티를 얼마나 커버하는지.

        Returns: 0.0 ~ 1.0 (커버된 GT 엔티티 비율)
        """
        top_indices = np.argsort(token_scores)[-k:][::-1]
        top_tokens = {token_strings[i].strip().lower() for i in top_indices}

        covered = 0
        for entity in ground_truth_entities:
            entity_tokens = entity.lower().split()
            if any(et in tok for tok in top_tokens for et in entity_tokens):
                covered += 1

        return covered / max(len(ground_truth_entities), 1)

    @staticmethod
    def prefix_waste_ratio(
        token_scores: np.ndarray,
        token_strings: List[str],
        prefixes: Optional[List[str]] = None,
    ) -> float:
        """Prefix 토큰에 할당된 attention 비율.
        높을수록 prefix에 attention이 낭비됨.

        Returns: 0.0 ~ 1.0
        """
        if prefixes is None:
            prefixes = AttentionAnalyzer.FIBO_PREFIXES

        prefix_attn = 0.0
        for i, tok in enumerate(token_strings):
            tok_lower = tok.strip().lower()
            if any(p in tok_lower for p in prefixes) or tok_lower in (":", "-"):
                prefix_attn += float(token_scores[i])

        total = float(token_scores.sum()) + 1e-10
        return prefix_attn / total

    @staticmethod
    def semantic_density(
        token_strings: List[str],
        prefixes: Optional[List[str]] = None,
    ) -> float:
        """의미 있는 토큰의 비율.
        높을수록 토큰 효율적.

        Returns: 0.0 ~ 1.0
        """
        if prefixes is None:
            prefixes = AttentionAnalyzer.FIBO_PREFIXES

        noise = 0
        for tok in token_strings:
            tok_lower = tok.strip().lower()
            if any(p in tok_lower for p in prefixes) or tok_lower in (":", "-", ""):
                noise += 1

        total = len(token_strings)
        return (total - noise) / max(total, 1)

    @staticmethod
    def per_head_entropy_stats(
        per_head_scores: Dict[int, np.ndarray],
    ) -> Dict[str, float]:
        """Head별 entropy 통계.

        Returns: {"mean": ..., "std": ..., "min": ..., "max": ...}
        """
        entropies = [
            AttentionAnalyzer.entropy(scores)
            for scores in per_head_scores.values()
        ]
        return {
            "mean": float(np.mean(entropies)),
            "std": float(np.std(entropies)),
            "min": float(np.min(entropies)),
            "max": float(np.max(entropies)),
        }
```

#### (4) 실험 드라이버 스크립트

```python
# src/attention_experiment.py

"""
LPG vs RDF Context → LLM Attention 비교 실험.

Usage (Colab A100):
    python src/attention_experiment.py \
        --model llama8b \
        --samples 100 \
        --output results/attention_experiment/
"""

class AttentionExperiment:
    """5가지 context 조건에 대한 attention 분석 실험."""

    CONDITIONS = ["lpg_structured", "lpg_natural", "rdf_raw", "rdf_cleaned", "no_context"]

    def __init__(self, config):
        self.llm = LocalLLMManager(config.model)
        self.evaluator = Evaluator()
        self.analyzer = AttentionAnalyzer()
        self.formatter = GraphFormatter()

    def build_context(self, condition, lpg_data, rdf_data):
        """조건별 context 텍스트 생성."""
        if condition == "lpg_structured":
            return self.formatter.format(lpg_data["nodes"], lpg_data["edges"],
                                         style="structured")
        elif condition == "lpg_natural":
            return self.formatter.format(lpg_data["nodes"], lpg_data["edges"],
                                         style="natural")
        elif condition == "rdf_raw":
            return self.formatter.format([], rdf_data["edges"], style="triple")
        elif condition == "rdf_cleaned":
            return self.formatter.format_rdf_cleaned(rdf_data["edges"])
        else:  # no_context
            return None

    def run_single(self, question, context, condition, entities):
        """단일 질문 × 단일 조건 실행."""
        response = self.llm.generate(
            question=question["text"],
            context=context,
            extract_attention=True,
            entity_names=entities,
        )

        # Attention metrics
        attn = response.attention_data
        metrics = {}
        if attn and len(attn.context_attention_scores) > 0:
            metrics["entropy"] = self.analyzer.entropy(attn.context_attention_scores)
            metrics["entity_coverage"] = self.analyzer.entity_coverage_at_k(
                attn.context_attention_scores, attn.context_tokens, entities
            )
            metrics["prefix_waste"] = self.analyzer.prefix_waste_ratio(
                attn.context_attention_scores, attn.context_tokens
            )
            metrics["semantic_density"] = self.analyzer.semantic_density(attn.context_tokens)

        # Answer quality
        quality = self.evaluator.evaluate_single(response.text, question["answer"])

        return {**metrics, **quality.__dict__, "condition": condition}

    def run(self):
        """전체 실험 루프."""
        samples = self.select_common_samples(n=100)
        results = []

        for q in samples:
            lpg_data = fetch_lpg_subgraph(q["id"])
            rdf_data = fetch_rdf_subgraph(q["id"])
            entities = extract_entity_names(lpg_data)

            for cond in self.CONDITIONS:
                context = self.build_context(cond, lpg_data, rdf_data)
                result = self.run_single(q, context, cond, entities)
                results.append(result)

                # Checkpoint every 10 questions
                ...

        self.save_results(results)
        self.generate_report(results)
```

---

## 5. 분석 계획

### 5.1 통계 검정

| 비교 | 검정 방법 | 귀무가설 |
|------|-----------|----------|
| C1 vs C3 (LPG vs RDF raw) | Paired Wilcoxon signed-rank | Entropy 차이 없음 |
| C3 vs C4 (RDF raw vs cleaned) | Paired Wilcoxon signed-rank | Prefix 제거가 entropy에 영향 없음 |
| C4 vs C1 (RDF cleaned vs LPG) | Paired Wilcoxon signed-rank | 포맷 차이가 없음 |
| Entity Coverage ↔ F1 | Spearman correlation | 상관 없음 |

### 5.2 시각화

#### (a) Attention Heatmap (Layer × Position)

```
y축: Layer 0~31
x축: Context 토큰 위치
값:  gen→ctx attention weight (평균)

LPG heatmap | RDF raw heatmap | RDF cleaned heatmap
```

#### (b) Per-Head Entropy Distribution

```
Boxplot: 각 조건별 32개 head의 entropy 분포
x축: Condition (C1~C5)
y축: Head Entropy (bits)
```

#### (c) Prefix Waste 분석

```
Stacked bar chart:
  각 조건별 attention 할당 비율
  [semantic tokens | structural tokens | prefix tokens]
```

#### (d) Entity Coverage vs Answer F1 Scatter

```
Scatter plot:
  x축: Entity Coverage@20
  y축: Token F1
  색상: Condition
  → 양의 상관관계 예측
```

### 5.3 예상 결과 및 해석

```
| Condition       | Tokens | Entropy↓ | PrefixWaste↓ | EntityCov↑ | F1↑   |
|-----------------|--------|----------|--------------|------------|-------|
| lpg_structured  | ~113   | low      | ~0%          | high       | ?     |
| lpg_natural     | ~130   | low      | ~0%          | high       | ?     |
| rdf_raw         | ~156   | HIGH     | ~50%         | medium     | ?     |
| rdf_cleaned     | ~90    | low      | ~0%          | high       | ?     |
| no_context      | 0      | N/A      | N/A          | N/A        | low   |
```

**가장 흥미로운 시나리오**: `rdf_cleaned`가 `lpg_structured`와 동등하거나 더 나은 성능을 보인다면, RDF의 정규화된 구조 + 자연어 변환이 최적 전략이 된다.

---

## 6. 실행 환경

| 항목 | 사양 |
|------|------|
| GPU | NVIDIA A100 40GB (Google Colab) |
| Model | Llama 3.1 8B Instruct (bf16, ~16GB) |
| Attention 추출 | `attn_implementation="eager"` (필수) |
| 예상 메모리 | ~20GB (model + attention tensors) |
| 예상 실행 시간 | ~2-3시간 (100 samples × 5 conditions × 32 layers) |

### 메모리 주의사항

Attention tensor 크기 (per forward pass):
```
32 layers × 32 heads × seq_len × seq_len × float32
= 32 × 32 × 2048 × 2048 × 4 bytes ≈ 16 GB (seq_len=2048일 때)
```

→ 전체 layer를 한 번에 추출하면 OOM 위험. **Layer를 배치로 나누거나**, `output_attentions`를 사용하되 즉시 슬라이싱 후 삭제하는 방식 필요. 현재 `AttentionExtractor`는 slicing 후 `del attentions; torch.cuda.empty_cache()`를 수행하므로 기본적으로 대응됨.

---

## 7. 구현 우선순위

| 순서 | 작업 | 예상 난이도 |
|------|------|------------|
| 1 | 공통 질문 샘플 선별 Neo4j 쿼리 | 쉬움 |
| 2 | `GraphFormatter.clean_rdf_uri()` + `format_rdf_cleaned()` | 쉬움 |
| 3 | `AttentionConfig`에 `aggregate_heads="none"` 옵션 추가 | 중간 |
| 4 | `AttentionExtractor` per-head 추출 확장 | 중간 |
| 5 | `AttentionAnalyzer` 새 클래스 구현 | 중간 |
| 6 | 실험 드라이버 `src/attention_experiment.py` | 높음 |
| 7 | 시각화 + 통계 분석 스크립트 | 중간 |

---

## 부록 A: 토큰 분해 상세

### Single fact: "NXP의 고객은 Apple이다"

**LPG (12 tokens)**:
```
Position: [0]  [1]   [2] [3] [4]  [5] [6]  [7] [8]    [9] [10]    [11]
Token:     -   HAS    _   C   UST  OM  ER   ->  Apple   (   Company  )
Role:      구조 관계   관계 관계 관계 관계 관계 구조 entity 구조 type    구조
```
- 의미 토큰: HAS, CUSTOMER (관계) + Apple (엔티티) + Company (타입) = 4/12 = **33%**
- 관계명이 영어 단어 → pre-training knowledge 활용 가능

**RDF (19 tokens)**:
```
Position: [0] [1] [2]  [3]     [4]  [5] [6] [7] [8] [9] [10] [11] [12] [13] [14] [15]     [16] [17] [18]
Token:     ex  :   Our  Company fib  o   -   f   nd  -   rel  -    rel  :    has  Customer  ex   :    Apple
Role:      pfx pfx ent  ent     pfx  pfx pfx pfx pfx pfx pfx  pfx  pfx  pfx rel  rel       pfx  pfx  ent
```
- FIBO prefix 토큰: [0,1,4,5,6,7,8,9,10,11,12,13,16,17] = **14/19 = 74%**
- 의미 토큰: Our, Company, has, Customer, Apple = **5/19 = 26%**

### Softmax 경쟁 시뮬레이션

질문 토큰 "distributor"가 context의 각 토큰에 attend할 때, softmax 분모에 기여하는 토큰 수:
- LPG: 12개 → 핵심 토큰 attention ≈ exp(score) / Σ₁₂
- RDF: 19개 → 핵심 토큰 attention ≈ exp(score) / Σ₁₉
- 동일 score 가정 시 LPG 대비 RDF의 attention weight = 12/19 ≈ **63%** (37% 감소)

## 부록 B: Neo4j 서브그래프 쿼리 참조

### finderlpg에서 질문별 서브그래프

```cypher
-- 노드
MATCH (e:Entity)
WHERE $qid IN e.question_ids
RETURN e.id, e.label, e.name

-- 엣지
MATCH (a:Entity)-[r]->(b:Entity)
WHERE $qid IN a.question_ids AND $qid IN b.question_ids
RETURN a.name AS src, a.label AS src_label, type(r) AS rel,
       b.name AS tgt, b.label AS tgt_label
```

### finderrdf에서 질문별 서브그래프

```cypher
-- 트리플 (question_id는 edge 속성)
MATCH (a:Resource)-[r]->(b:Resource)
WHERE r.question_id = $qid
RETURN a.uri AS subject, type(r) AS predicate, b.uri AS object
```

> **주의**: 일부 질문은 LPG에만 데이터가 있고 RDF에는 없음. 실험 샘플은 반드시 양쪽 교집합에서 선별해야 함.
