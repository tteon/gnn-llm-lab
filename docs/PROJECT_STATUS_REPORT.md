# FinDER Graph RAG Lab — 프로젝트 현황 보고서

> **작성일**: 2026-02-15
> **목적**: 프로젝트 맥락 파악용 종합 현황 정리

---

## 1. 프로젝트 개요

### 무엇을 하는 프로젝트인가?

금융 QA 데이터셋(FinDER)에서 **Knowledge Graph를 자동 구축**하고, 이를 **텍스트로 직렬화하여 LLM에 soft prompt**로 주입했을 때 답변 품질이 어떻게 달라지는지를 체계적으로 비교하는 실험 랩이다.

핵심 질문: **"LLM에 그래프 컨텍스트를 텍스트로 넣으면 답변이 나아지는가? LPG vs RDF 중 어떤 표현이 더 효과적인가?"**

### 접근 방식

```
[데이터] HuggingFace FinDER (5,703 금융 QA)
    ↓ LLM 엔티티 추출 (GPT-4o-mini)
[KG 구축] LPG (13,920 nodes, 18,892 edges) + RDF (12,365 nodes, 12,609 edges)
    ↓ Neo4j 적재
[실험] 5가지 컨텍스트 조건 × LLM 생성 → 메트릭 비교
    ↓ Opik 대시보드
[평가] EM, Token F1, ROUGE, BERTScore + LLM-as-Judge
```

### 기술 스택

| 구성요소 | 기술 |
|---------|------|
| 언어 / 패키지 | Python 3.10, UV |
| 그래프 DB | Neo4j (DozerDB 5.26.3, Docker) |
| LLM | Llama 3.1 8B Instruct (vLLM 서빙) |
| KG 추출 | GPT-4o-mini (OpenAI API) |
| 임베딩 | all-MiniLM-L6-v2 (384차원, few-shot용) |
| 실험 추적 | Opik (self-hosted Docker) |
| 평가 | Custom Evaluator + Opik LLM-as-Judge |

---

## 2. 재현 가능한 파이프라인

`make pipeline` 한 줄로 전체 파이프라인 실행 가능:

```
Stage 1: make download     → HuggingFace → data/raw/FinDER.parquet (5,703 rows)
Stage 2: make build-kg     → LLM extraction → FinDER_KG_Merged.parquet (+KG 컬럼)
Stage 3: make load-neo4j   → Parquet → Neo4j (finderlpg + finderrdf)
Stage 4: make experiment   → Opik 실험 실행 → Dashboard
```

### Stage 2: KG 구축 상세

- **입력**: 각 QA 쌍의 `text` 필드
- **처리**: 카테고리별 프롬프트 라우팅
  - `Financials` → FIBO 온톨로지 프롬프트 (COMPANY_NAME, FINANCIAL_TERM, CURRENCY 등)
  - 나머지 → 범용 프롬프트 (Organization, Person, Location 등)
- **출력**: `lpg_nodes`, `lpg_edges`, `rdf_triples` (JSON 컬럼)
- **비용**: ~17K API 호출, ~30분 (500 RPM)
- **체크포인트**: `data/intermediate/kg_extraction.jsonl` (중단 후 재개 가능)
- **Opik 트레이싱**: `--no-opik` 플래그로 제어 (2026-02-15 추가)

### Neo4j 데이터베이스

| DB | 유형 | 노드 | 엣지 | 비고 |
|----|------|------|------|------|
| `finderlpg` | LPG | 13,920 | 18,892 | 100+ 엔티티 타입, 속성 포함 |
| `finderrdf` | RDF | 12,365 | 12,609 | FIBO URI 기반 트리플 |

**공통 질문 수**: 1,332개 (LPG ∩ RDF, 엔티티/트리플 ≥ 3 필터)

---

## 3. 실험 설계

### 3축 실험 매트릭스

| 축 | 값 | 설명 |
|----|----|------|
| **Models** | llama8b, mixtral, qwen_moe, llama70b | 현재 llama8b 중심 |
| **Contexts** | none, text, lpg, rdf, lpg_rdf | 5가지 컨텍스트 조건 |
| **Few-shot** | zeroshot, fewshot | centroid-nearest per-category |

### 5가지 컨텍스트 조건

| 조건 | 데이터 흐름 | 설명 |
|------|-----------|------|
| `none` | Question → LLM | 컨텍스트 없이 LLM만 |
| `text` | Question + References → LLM | 원본 텍스트 참조 (Text RAG) |
| `lpg` | Neo4j(finderlpg) → GraphFormatter → LLM | LPG 서브그래프를 텍스트로 |
| `rdf` | Neo4j(finderrdf) → format_rdf_cleaned → LLM | RDF 트리플을 텍스트로 |
| `lpg_rdf` | Neo4j(both) → format_combined → LLM | LPG + RDF 결합 (60:40 예산) |

### GraphFormatter 직렬화 스타일

| 스타일 | 예시 |
|--------|------|
| **structured** (기본) | `[Company] Apple Inc: tech company` + `Tim Cook --[CEO_OF]--> Apple Inc` |
| **natural** | `Apple Inc is a technology company. Tim Cook is the CEO of Apple Inc.` |
| **triple** | `(Apple Inc, founded_in, 1976)` |
| **csv** | CSV 테이블 형식 |

### LLM 생성 설정

| 파라미터 | 기본값 |
|---------|--------|
| `max_new_tokens` | 256 |
| `temperature` | 0.0 (결정론적) |
| `max_hops` (Neo4j) | 2 |
| `max_context_nodes` | 30 |
| `max_context_edges` | 50 |
| `soft_prompt_format` | structured |

### Opik Agent Trace 구조 (2026-02-15 리팩토링)

**KG Build 파이프라인** (`scripts/build_kg.py`):
```
kg_build_pipeline (per sample)
├── entity_extraction      ← LLM API 호출 (fibo/base 프롬프트)
├── entity_linking         ← LLM API 호출 (관계 추출)
└── rdf_conversion         ← 결정론적 FIBO URI 매핑
```

**실험 파이프라인** (`src/opik_experiment.py`):
```
graph_rag_pipeline (per question)
├── router_agent               ← 컨텍스트 라우팅 결정
├── lpg_retrieval_agent        ← Neo4j finderlpg 조회 (조건부)
├── rdf_retrieval_agent        ← Neo4j finderrdf 조회 (조건부)
└── answer_generation_agent    ← LLM 생성 + few-shot
```

---

## 4. 지금까지의 실험 결과

### 4.1 KV Cache Offloading 실험 (2026-02-08)

**목적**: vLLM prefix caching으로 동일 프롬프트 재사용 시 속도 향상 확인

**설계**: 5 조건 × 50 질문 = 250 실행 (cold start + warm restart)

| 지표 | 결과 |
|------|------|
| Cold Latency | 7.80 ± 1.15 sec |
| Warm Latency | 7.75 ± 1.10 sec |
| Speedup | **1.006× (효과 없음)** |
| Cache Hit Rate | **0.0** |
| Token F1 | 0.189 |
| ROUGE-L | 0.181 |
| Exact Match | **0.0** (CoT 출력이 ground truth와 불일치) |

**핵심 발견**: `gpu_prefix_cache_hit_rate = 0.0` — vLLM `--enable-prefix-caching` 플래그 누락 또는 CoT 패턴 간섭으로 prefix caching이 작동하지 않음.

### 4.2 초기 Graph RAG 실험 (2026-02-07)

**위치**: `results/experiments/20260207_073729_results_final.csv`

4가지 조건 비교 (llm_only, text_rag, graph_lpg, graph_rdf)로 초기 탐색 실험 수행. 본격적인 정량 비교는 아직 미완료.

### 4.3 GNN/KGE 학습 (Legacy)

7개 모델 학습 완료 (총 138MB 체크포인트):

| 모델 유형 | 모델 | 크기 |
|----------|------|------|
| GNN | GAT, GCN, Graph Transformer | 각 21MB |
| KGE | TransE, DistMult, ComplEx, RotatE | 각 19MB |

**현재 상태**: `src/_legacy/`로 이동, soft prompt 파이프라인에서는 사용하지 않음.

---

## 5. 아직 실행하지 않은 것 (본실험)

### 계획된 Full Experiment

| 항목 | 값 |
|------|-----|
| 모델 | llama8b (1차) |
| 컨텍스트 | none, text, lpg, rdf, lpg_rdf (5개) |
| Few-shot | zeroshot, fewshot (2개) |
| 질문 수 | 1,332 (공통 질문) |
| **총 실험** | **10 experiments (1 × 5 × 2)** |
| **총 평가** | **13,320 evaluations** |
| 예상 시간 | ~11시간 (llama8b) |

### 실행 명령

```bash
# Phase 1: Graph contexts only (핵심 비교, judge 없이)
uv run python src/opik_experiment.py \
    --models llama8b \
    --contexts none lpg rdf lpg_rdf \
    --sample-size 1332 \
    --no-judge --no-bertscore

# Phase 2: Full matrix + LLM-as-Judge
uv run python src/opik_experiment.py \
    --models llama8b \
    --contexts none text lpg rdf lpg_rdf \
    --few-shot \
    --sample-size 1332 \
    --judge-model gpt-4o-mini
```

---

## 6. Known Issues & 개선 백로그

### Critical (결과 직접 영향)

| # | 이슈 | 영향 | 해결 방안 |
|---|------|------|----------|
| 1 | RDF URI 프리픽스 토큰 낭비 | ~57.8% 토큰이 URI에 소비 | `format_rdf_cleaned()` 적용 (구현 완료, 실험 미적용) |
| 2 | LPG "None" 노드 낭비 | ~10% 토큰 낭비 | Cypher WHERE 필터 추가 |
| 3 | Hub 노드 지배 | "The Company" (degree=3,122)가 대부분 서브그래프에 포함 | Hub pruning 또는 PCST 기반 선택 |

### High (인프라)

| # | 이슈 | 상태 |
|---|------|------|
| 4 | KV Cache prefix 재사용 불가 | vLLM 설정 확인 필요 |
| 5 | Exact Match 항상 0.0 | CoT 응답 파싱 로직 필요 |

### Medium (기능)

| # | 이슈 | 상태 |
|---|------|------|
| 6 | Attention 실험 미실행 | 코드 완료, A100 GPU 필요 |
| 7 | Hard Prompt 실험 미구현 | GNN/KGE → LLM 직접 주입 |
| 8 | rdf_cleaned 조건 미테스트 | `format_rdf_cleaned()` 실험 적용 필요 |

---

## 7. 프로젝트 구조 요약

```
gnnllm_lab/
├── scripts/                     # 파이프라인 스크립트
│   ├── download_dataset.py      # Stage 1: HF 다운로드
│   └── build_kg.py              # Stage 2: LLM KG 추출 (+ Opik trace)
├── src/
│   ├── load_finder_kg.py        # Stage 3: Neo4j 적재
│   ├── run_experiment.py        # 3축 실험 러너
│   ├── opik_experiment.py       # Opik 통합 실험 (agent trace)
│   ├── utils/                   # 핵심 유틸리티 (11 모듈)
│   │   ├── config.py            # ExperimentConfig, FewShotConfig
│   │   ├── local_llm.py         # LocalLLMManager + MODEL_REGISTRY
│   │   ├── formatting.py        # GraphFormatter (structured/natural/triple/csv)
│   │   ├── neo4j_client.py      # Neo4j 클라이언트 (retry 포함)
│   │   ├── evaluation.py        # Evaluator (EM, F1, ROUGE, BERTScore)
│   │   ├── few_shot.py          # FewShotSelector (centroid-nearest)
│   │   └── ...                  # logging, exceptions, reproducibility
│   └── _legacy/                 # GNN/KGE/attention 코드 보존 (11 파일)
├── prompts/                     # 5 YAML 프롬프트 템플릿
├── data/
│   ├── raw/                     # FinDER.parquet, FinDER_KG_Merged.parquet
│   ├── processed/               # common_question_ids.json (1,332개)
│   └── intermediate/            # 체크포인트
├── results/                     # 실험 결과
│   ├── experiments/             # CSV 결과 파일
│   ├── kvcache_experiment/      # KV cache 실험 (summary + metrics)
│   └── checkpoints/             # GNN/KGE 모델 체크포인트 (138MB)
├── notebooks/                   # 분석 노트북 (4 active + 1 legacy)
├── docs/                        # 문서 (11 MD, 4,802 lines)
│   ├── design/                  # 실험 설계서
│   ├── analysis/                # 분석 보고서
│   └── reference/               # Neo4j 스키마, cuGraph 분석
├── Makefile                     # pipeline, experiment, lint, test
├── CLAUDE.md                    # AI 어시스턴트 지침
└── KNOWN_ISSUES.md              # 개선 백로그 (11항목, 6 resolved)
```

---

## 8. 현재 상태 한줄 요약

**인프라 완성, 본실험 대기 중.**

- 재현 가능한 E2E 파이프라인 (`make pipeline`) 구축 완료
- KG 구축 (LPG + RDF) 및 Neo4j 적재 완료
- Opik agent trace 아키텍처 구현 완료 (4-agent span)
- 1,332개 공통 질문으로 5 컨텍스트 × 2 shot = 10 실험 실행 준비 완료
- KV cache 탐색 실험은 완료했으나 prefix caching 미작동 이슈 확인
- **남은 작업: 본실험 실행 (~11시간) → 결과 분석 → 논문/보고서**
