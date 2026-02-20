# G-Retrieval 비교 실험 설계

## 목적

FinDER KG의 dual-graph (LPG + RDF) 서브그래프에 대해 서로 다른 그래프 인코더가 답변 정보를 얼마나 잘 capture하는지 비교한다.

## 모델 선택 근거

### GAT (LPG)

| 항목 | 설명 |
|------|------|
| 입력 | 384d SentenceTransformer node features + edge_index |
| 구조 | 2-layer GATConv (4 heads) + LayerNorm + global_mean_pool |
| 출력 | 384d graph-level embedding |
| 선택 이유 | LPG의 노드 features가 이미 semantic하게 풍부 (all-MiniLM-L6-v2); attention으로 중요 이웃에 집중 |

기존 `MessagePassingGNN`의 `x.mean(dim=0)` → `global_mean_pool(x, batch)`로 변경하여 mini-batch 지원.

### TransE (RDF)

| 항목 | 설명 |
|------|------|
| 이론 | Translation: h + r ≈ t |
| 강점 | **비대칭 관계** 모델링 (OWNS, REPORTED, COMPLY_WITH) |
| 약점 | Margin hyperparameter 민감, 대칭 관계 모델링 한계 |
| FinDER 결과 | MRR=0.053, Hits@10=11.0% (link prediction) |

### DistMult (RDF)

| 항목 | 설명 |
|------|------|
| 이론 | Bilinear: score = h · diag(r) · t |
| 강점 | 학습 안정적, 실증적으로 FinDER에서 TransE보다 우수 |
| 약점 | 대칭 가정 — OWNS(A,B)와 OWNS(B,A) 구분 불가 |
| FinDER 결과 | MRR=0.077, Hits@10=14.2% (link prediction) |

### 왜 둘 다?

FinDER KG는 방향성 관계(OWNS, REPORTED)와 대칭적 관계(RELATED_TO) 모두 포함.
- DistMult: 실증적 우위 (MRR +46%)
- TransE: 이론적 장점 (비대칭 관계)
- 카테고리별로 어떤 모델이 유리한지 분석 가능

## 학습 전략

### GAT: Link Prediction (Self-supervised)

```
Forward:  x, edge_index → GAT layers → node embeddings z
Positive: z[src] · z[dst] → high score
Negative: negative_sampling() → z[src'] · z[dst'] → low score
Loss:     BCE(pos, 1) + BCE(neg, 0)
```

- Epochs: 50, lr=1e-3, weight_decay=1e-5
- Batch: DualGraphBatch (32 graphs/batch)
- Val: Sampled MRR + Hits@10 (매 5 epoch)
- Best model: Val MRR 기준 early stopping

### KGE: Triple Scoring

```
Forward:  (h, r, t) → KGE loss (built-in negative sampling)
          TransE:   margin ranking loss
          DistMult: negative sampling loss
```

- Epochs: 100, lr=1e-2, batch_size=512
- 전체 train triples를 수집하여 global KGE 학습
- Val: Sampled MRR + Hits@10 (매 10 epoch)

## Graph Embedding 생성

학습 후 per-question graph embedding 추출:

| 모델 | 방법 |
|------|------|
| GAT | `forward(x, edge_index, batch)` → `global_mean_pool` → [B, 384] |
| TransE/DistMult | triple별 `h_emb + r_emb` → `scatter(reduce='mean', by=graph)` → [B, 384] |

모든 embedding은 L2 normalize 후 사용.

## 평가 방법

### 1. Graph → Answer Retrieval

질문의 graph embedding으로 정답 answer의 text embedding을 검색하는 task.

```
Query:  graph_emb[i]  (384d, from GAT/TransE/DistMult)
Corpus: answer_emb[j] (384d, from SentenceTransformer)
Score:  cosine similarity
Ground truth: i == j (diagonal)
```

**Metrics:**
- **MRR** (Mean Reciprocal Rank): 정답의 평균 역순위
- **Recall@1/5/10**: Top-K에 정답 포함 비율

**Baseline:**
- Question text embedding → Answer text embedding (그래프 없이 텍스트만)

### 2. Category-wise Breakdown

8개 FinDER 카테고리별 retrieval 성능 비교:
- 어떤 카테고리에서 LPG(GAT) vs RDF(KGE) 우위?
- 카테고리별 최적 모델 추천

### 3. Graph Size Effect

서브그래프 크기(노드 수, 엣지 수)와 retrieval 성능의 상관관계:
- Spearman correlation
- Binned scatter plot

### 4. TransE vs DistMult Deep Dive

- Loss curve 비교
- Per-sample win/loss 분석
- t-SNE embedding space 시각화

## 예상 결과 해석 가이드

| 결과 패턴 | 해석 |
|-----------|------|
| GAT >> KGE | LPG의 사전학습된 node features가 지배적; KGE는 entity가 너무 많아 학습 불충분 |
| DistMult > TransE | 대칭 가정이 성능 저하 없이 학습 안정성 제공; FinDER의 관계가 생각보다 대칭적 |
| TransE > DistMult (특정 카테고리) | 해당 카테고리에 방향성 관계(OWNS, REPORTED)가 많음 |
| 큰 그래프에서 성능 ↑ | 더 많은 context가 답변 정보를 더 잘 capture |
| 큰 그래프에서 성능 ↓ | 노이즈 증가; mean pooling이 중요 정보를 희석 |
| Question baseline >> Graph | 그래프 정보가 텍스트 대비 추가 가치 없음; soft-prompt이 더 효과적일 수 있음 |
| Graph >> Question baseline | 그래프 구조가 텍스트에 없는 관계 정보를 capture |

## 데이터 의존성

| 파일 | 용도 | 크기 |
|------|------|------|
| `data/processed/finder_pyg/processed/train.pt` | Train split (2,030 samples) | 28 MB |
| `data/processed/finder_pyg/processed/val.pt` | Val split (251 samples) | 3.3 MB |
| `data/processed/finder_pyg/processed/test.pt` | Test split (261 samples) | 3.4 MB |
| `data/processed/finder_pyg/processed/vocab.pt` | Entity/relation vocabularies | 827 KB |
| `data/processed/finder_pyg/processed/metadata.json` | Split info, vocab sizes | 382 B |

Google Drive 캐시 경로: `gnnllm_lab_data/finder_pyg/processed/`

## Colab 환경

- GPU: T4 (16GB) 또는 A100
- 예상 학습 시간: GAT ~5min, KGE ~10min (T4 기준)
- Setup: `bash scripts/colab_setup.sh`
- Notebook: `notebooks/g_retrieval_comparison.ipynb`
