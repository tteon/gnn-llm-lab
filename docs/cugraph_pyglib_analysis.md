# cuGraph / pyg-lib / PyG 가속 기술 분석 보고서

> GNN+LLM Lab 프로젝트 기준 (14K 노드, RTX 3070, PyTorch 2.10.0+cu128, PyG 2.7.0)

## 1. Executive Summary

이 프로젝트의 그래프 규모(LPG 13,920 노드 / RDF 12,365 노드)에서는 billion-edge 타겟의 cuGraph-PyG는 불필요하다. 대신 **torch.compile()** (2-3x 가속, 코드 1줄)과 **pyg-lib 설치** (scatter ops 최적화)가 가장 실용적이다. bfloat16 AMP는 메모리 절약에 유용하지만 현재 모델 크기(~2M params)에서는 필수는 아니다.

| 가속 기술 | 적용 가치 | 기대 효과 | 필요 작업 |
|-----------|-----------|-----------|-----------|
| **torch.compile()** | **높음** | 2-3x 학습 가속 | 1줄 코드 |
| **pyg-lib 설치** | **중간** | scatter ops 가속, neighbor sampling 최적화 | 패키지 설치 |
| **bfloat16 AMP** | **중간** | 메모리 ~50% 절약, 약간의 속도 향상 | ~10줄 코드 |
| **torch-scatter 설치** | **낮음** | scatter_mean ~2x 가속 (PyG fallback 대비) | 패키지 설치 |
| **cuGraph-PyG** | **낮음** | 14K 노드에서 이점 없음 | 해당 없음 |
| **CuGraphGATConv** | **불가** | RAPIDS 25.02에서 제거됨 | 불가 |

---

## 2. pyg-lib

### 역할
[pyg-lib](https://github.com/pyg-team/pyg-lib)은 PyG의 핵심 low-level 연산을 최적화하는 C++/CUDA 확장 라이브러리.

- **Neighbor sampling** 10-15x 가속 (CPU/GPU 커널)
- **Heterogeneous GNN** grouped matmul (CUTLASS 기반)
- **Message passing** 최적화

### PyG와의 통합
PyG 2.7.0에서 자동 감지 — `import pyg_lib` 가능하면 자동으로 최적화된 경로를 사용. 코드 변경 불필요.

### 설치

```bash
pip install pyg_lib -f https://data.pyg.org/whl/torch-2.10.0+cu128.html
```

### 현재 환경 호환성

| 항목 | 현재 값 | pyg-lib 지원 |
|------|---------|-------------|
| PyTorch | 2.10.0 | 확인 필요 (2.8.0까지 공식 휠 확인됨) |
| CUDA | 12.8 | PyTorch 2.7+ 부터 cu128 휠 제공 |
| Python | 3.10 | 지원 |

> **참고**: PyTorch 2.10.0은 최신 버전으로, pyg-lib 휠이 아직 대응하지 않을 수 있음. `pip install` 시도 후 실패하면 소스 빌드 필요.

### 이 프로젝트에서의 이점
- 현재 `torch_scatter`, `torch_sparse` 미설치 상태 → PyG가 순수 PyTorch fallback 사용 중
- pyg-lib 설치만으로 scatter 연산 가속 가능
- 14K 노드 규모에서는 극적 차이는 아니지만, 반복 실험 시 누적 효과 있음

---

## 3. cuGraph-PyG

> 소스: [rapidsai/cugraph-gnn](https://github.com/rapidsai/cugraph-gnn/tree/main/python/cugraph-pyg) (v26.04, Apache-2.0)

### 3.1 역할과 아키텍처

cuGraph-PyG는 NVIDIA RAPIDS의 **데이터 파이프라인 가속기**. PyG의 4가지 핵심 인터페이스를 GPU 네이티브로 구현하여, GNN 학습에서 데이터 로딩/샘플링 병목을 제거한다. **GNN 레이어(conv)는 포함하지 않는다** — `nn/` 디렉토리 자체가 존재하지 않으며, 연산은 네이티브 PyG 레이어(`GCNConv`, `GATConv`, `FastRGCNConv` 등)를 그대로 사용.

```
┌─────────────────────────────────────────────────────────────┐
│                     cuGraph-PyG 아키텍처                      │
├──────────────┬──────────────┬───────────────┬───────────────┤
│  Data Layer  │ Sampler Layer│  Loader Layer │  Tensor Layer │
├──────────────┼──────────────┼───────────────┼───────────────┤
│ GraphStore   │ BaseSampler  │ NodeLoader    │ DistTensor    │
│ FeatureStore │ Distributed  │ NeighborLoader│ DistEmbedding │
│              │ Neighbor     │ LinkLoader    │ DistMatrix    │
│              │ Sampler      │ LinkNeighbor  │               │
│              │              │ Loader        │               │
├──────────────┴──────────────┴───────────────┴───────────────┤
│              pylibcugraph (C++ CUDA backend)                 │
│              pylibwholegraph (distributed storage)           │
└─────────────────────────────────────────────────────────────┘
                        ↓ 사용
┌─────────────────────────────────────────────────────────────┐
│         PyG 네이티브 GNN 레이어 (GCNConv, GATConv 등)         │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 핵심 발견: cugraph-ops 제거

**RAPIDS 25.02에서 `cugraph-ops` 패키지가 완전 제거됨** ([RSN0041](https://docs.rapids.ai/notices/rsn0041/))

영향받는 컴포넌트:
- `CuGraphGATConv` — GATConv의 GPU-fused 대체
- `CuGraphSAGEConv` — GraphSAGE의 GPU-fused 대체
- `CuGraphRGCNConv` — RGCNConv의 GPU-fused 대체
- `libcugraphops`, `pylibcugraphops` — 모두 삭제

추가로 cuGraph-DGL은 release 25.08에서 완전 제거됨. cuGraph-PyG가 유일한 지원 프레임워크.

### 3.3 PyG 인터페이스 매핑

cuGraph-PyG는 PyG의 인터페이스를 2가지 방식으로 구현:

| PyG 인터페이스 | cuGraph-PyG 구현 | 통합 방식 |
|----------------|-----------------|-----------|
| `torch_geometric.data.GraphStore` | `cugraph_pyg.data.GraphStore` | 조건부 상속 (PyG 있으면 상속, 없으면 `object`) |
| `torch_geometric.data.FeatureStore` | `cugraph_pyg.data.FeatureStore` | 조건부 상속 |
| `torch_geometric.loader.NeighborLoader` | `cugraph_pyg.loader.NeighborLoader` | **Duck typing** (상속 없음) |
| `torch_geometric.loader.NodeLoader` | `cugraph_pyg.loader.NodeLoader` | Duck typing |
| `torch_geometric.loader.LinkLoader` | `cugraph_pyg.loader.LinkLoader` | Duck typing |
| `torch_geometric.loader.LinkNeighborLoader` | `cugraph_pyg.loader.LinkNeighborLoader` | Duck typing |

> Duck typing 패턴: Loader 클래스들은 PyG 클래스를 상속하지 않지만 동일한 인터페이스를 구현. PyG가 하드 의존성이 아닌 선택적 import (`import_optional`)로 처리됨.

### 3.4 핵심 클래스 상세

#### GraphStore — GPU 그래프 저장소

`pylibcugraph.SGGraph` (단일 GPU) 또는 `pylibcugraph.MGGraph` (멀티 GPU)를 백엔드로 사용.

```python
# cugraph_pyg/data/graph_store.py
class GraphStore(torch_geometric.data.GraphStore):  # 조건부 상속
    def _put_edge_index(self, edge_index, edge_attr): ...  # 분산 동기화 포함
    def _get_edge_index(self, edge_attr): ...              # COO/CSR/CSC 형식 반환
    def _graph(self): ...                                   # Lazy SGGraph/MGGraph 생성

    @property
    def is_multi_gpu(self): ...       # 멀티 GPU 모드 여부
    @property
    def is_homogeneous(self): ...     # 동질 그래프 여부
    @property
    def _vertex_offsets(self): ...    # 정점 타입별 글로벌 오프셋
```

#### FeatureStore — WholeGraph 분산 특성 저장소

**WholeGraph WholeMemory** 백엔드를 사용하여 데이터 복제 없는 분산 특성 저장.

```python
# cugraph_pyg/data/feature_store.py
class FeatureStore(torch_geometric.data.FeatureStore):  # 조건부 상속
    def __init__(self, memory_type=None, location="cpu"):
        # WholeMemory 백엔드 선택 (NVLink 감지 포함)
    def _put_tensor(self, tensor, attr): ...    # DistTensor/DistEmbedding 저장
    def _get_tensor(self, attr): ...            # 인덱스 슬라이싱 지원
```

#### NeighborLoader — GPU 네이버 샘플링

```python
# cugraph_pyg/loader/neighbor_loader.py
class NeighborLoader(NodeLoader):
    def __init__(
        self,
        data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
        num_neighbors: Union[List[int], Dict[EdgeType, List[int]]],
        input_nodes: InputNodes = None,
        batch_size: int = 16,
        replace: bool = False,
        subgraph_type: str = "directional",
        disjoint: bool = False,
        temporal_strategy: str = "uniform",
        time_attr: Optional[str] = None,
        weight_attr: Optional[str] = None,
        compression: Optional[str] = None,
        local_seeds_per_call: Optional[int] = None,
        ...
    ): ...
```

내부적으로 `DistributedNeighborSampler`를 생성하여 `pylibcugraph` C++ 샘플링 프리미티브를 호출.

#### DistributedNeighborSampler — 핵심 샘플링 엔진

8가지 샘플링 구성을 `_func_table`로 매핑:
- `(homogeneous/heterogeneous) × (uniform/biased) × (temporal/non-temporal)` → 전용 C++ 함수

```python
# cugraph_pyg/sampler/distributed_sampler.py
class DistributedNeighborSampler(BaseDistributedSampler):
    BASE_VERTICES_PER_BYTE = 0.1107662486009992  # GPU 메모리 기반 자동 튜닝 상수

    def __init__(
        self, graph, *,
        local_seeds_per_call=None,  # None이면 GPU 메모리 기반 자동 계산
        fanout=[-1],                # -1 = 전체 이웃
        compression="COO",          # COO 또는 CSC
        biased=False,
        heterogeneous=False,
        temporal=False,
        ...
    ): ...

    def sample_from_nodes(self, nodes, *, batch_size=16, ...): ...
    def sample_from_edges(self, edges, *, batch_size=16, ...): ...
```

### 3.5 전형적 워크플로우

#### 노드 분류 (GCN, 멀티 GPU)

`examples/gcn_dist_mnmg.py` 기반:

```python
import cugraph_pyg

# 1. GPU 스토어 생성
feature_store = cugraph_pyg.data.FeatureStore()
graph_store = cugraph_pyg.data.GraphStore()

# 2. 데이터 로드 (WholeGraph 백엔드로 분산 저장)
feature_store[("node", "x", None)] = node_features       # 노드 특성
feature_store[("node", "y", None)] = node_labels          # 노드 레이블
graph_store[("node", "rel", "node")] = edge_index_coo     # 엣지 (COO 형식)

# 3. GPU NeighborLoader 생성
loader = cugraph_pyg.loader.NeighborLoader(
    (feature_store, graph_store),
    num_neighbors=[fan_out] * num_layers,  # e.g., [25, 10] for 2-layer
    input_nodes=train_nodes,
    batch_size=batch_size,
)

# 4. 표준 PyG 학습 루프 (GNN 레이어는 네이티브 PyG)
model = GCN(in_channels, hidden_channels, out_channels, num_layers)
for batch in loader:
    out = model(batch.x, batch.edge_index)
    loss = F.cross_entropy(out[batch.train_mask], batch.y[batch.train_mask])
    loss.backward()
    optimizer.step()
```

#### 링크 예측 (R-GCN, 멀티 GPU)

`examples/rgcn_link_class_mnmg.py` 기반:

```python
# 이종 그래프 링크 예측
feature_store[("n", "emb", None)] = node_embeddings  # Xavier 초기화 학습 임베딩
feature_store[("n", "e", "n"), "edge_type", None] = edge_types.int()
graph_store[("n", "e", "n")] = edge_index

# LinkNeighborLoader로 엣지 배치 샘플링
loader = cugraph_pyg.loader.LinkNeighborLoader(
    (feature_store, graph_store),
    num_neighbors=[fan_out] * num_layers,
    edge_label_index=train_edges,
    batch_size=16384,
    shuffle=True,
)

# R-GCN + GAE loss (PyG FastRGCNConv 사용)
model = RGCNEncoder(num_nodes, hidden_channels, num_relations)
for batch in loader:
    z = model(batch.edge_index, batch.edge_type)
    loss = model.recon_loss(z, batch.edge_label_index)
```

#### 멀티 GPU 실행

```bash
# torchrun으로 NCCL 기반 분산 실행
torchrun --nproc_per_node=4 examples/gcn_dist_mnmg.py --dataset ogbn-papers100M
```

### 3.6 의존성 요구사항

| 의존성 | 버전 제약 | 비고 |
|--------|-----------|------|
| **Python** | >= 3.11 | 3.10 미지원 |
| **torch-geometric** | >= 2.5, < 2.8 | PyG 2.7.0 호환 |
| **torch** (test) | >= 2.9.0 | |
| **cupy-cuda13x** | >= 13.6.0 | CUDA 13.x 필요 |
| **pylibcugraph** | == 26.4.* | RAPIDS C++ 백엔드 |
| **pylibwholegraph** | == 26.4.* | 분산 텐서 저장 |

### 3.7 이 프로젝트 기준 평가

| 기능 | 이 프로젝트 적합성 | 이유 |
|------|-------------------|------|
| GPU graph sampling | **불필요** | 14K 노드 전체를 메모리에 올림 (full-graph training, mini-batch 불필요) |
| LinkNeighborLoader | **불필요** | 현재 전체 엣지 사용, 샘플링 병목 없음 |
| WholeGraph 분산 저장 | **불필요** | 단일 GPU, 그래프 데이터 ~23MB |
| 멀티 GPU 학습 | **불필요** | RTX 3070 단일 카드 |
| Fused convolutions | **불가** | RAPIDS 25.02에서 cugraph-ops 제거 |

### 3.8 conda/UV 생태계 충돌

cuGraph-PyG는 RAPIDS conda 생태계에 강하게 의존:
- `pylibcugraph`, `pylibwholegraph`는 `conda install -c rapidsai` 로 설치
- Python >= 3.11 요구 (이 프로젝트는 Python 3.10)
- UV의 pip 기반 환경과 RAPIDS conda 패키지 충돌 가능

### 3.9 cuGraph-PyG가 유의미한 시나리오

참고용으로, cuGraph-PyG가 실질적 이점을 주는 규모:

| 규모 | 노드 수 | 엣지 수 | cuGraph 이점 |
|------|---------|---------|-------------|
| 이 프로젝트 | 14K | 19K | 없음 |
| 중간 규모 (Cora, Citeseer) | 2K-20K | 5K-100K | 없음 |
| OGB-level (ogbn-arxiv) | 170K | 1.2M | 미미함 |
| 대규모 (ogbn-papers100M) | **100M+** | **1B+** | **높음** |
| 산업 규모 (추천, 소셜) | **1B+** | **10B+** | **매우 높음** |

**결론: 이 프로젝트(14K 노드)에서는 cuGraph-PyG 도입이 불필요하며, 의존성 복잡성만 증가시킨다. billion-edge 규모로 확장하거나 멀티 GPU 학습이 필요할 때 도입을 고려.**

---

## 4. torch-scatter / torch-sparse / torch-cluster

### PyG 2.7에서의 의존성 상태

PyG 2.3부터 scatter 연산이 `torch_geometric.utils.scatter`로 내재화됨. PyG 2.7.0에서 이들 패키지는 **선택적(optional)**:

| 패키지 | 역할 | 미설치 시 |
|--------|------|-----------|
| `torch-scatter` | 최적화된 scatter 연산 (sum, mean, max) | PyTorch 네이티브 fallback (`torch.scatter_reduce`) |
| `torch-sparse` | SparseTensor 형식, sparse matmul | 일반 `edge_index` 형식 사용 |
| `torch-cluster` | kNN, radius graph 등 | 일부 기능 사용 불가 |

### 성능 차이

```
scatter_mean 벤치마크 (합성 100K 노드):
  torch-scatter:          ~1.2ms
  PyTorch native fallback: ~2.5ms  (약 2x 느림)
```

### 현재 환경 상태

```
pyg_lib:       NOT installed
torch_scatter: NOT installed
torch_sparse:  NOT installed
```

→ 현재 모든 scatter 연산이 PyTorch fallback으로 실행 중.

### 권장 사항
- `torch-scatter` 설치가 가장 효과적 (scatter 연산은 GNN 학습에서 가장 빈번)
- `torch-sparse`는 SparseTensor 형식을 사용할 때만 필요
- `torch-cluster`는 이 프로젝트에서 불필요 (kNN graph 미사용)

```bash
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.10.0+cu128.html
```

---

## 5. torch.compile() + PyG

### 개요
PyG 2.5+에서 `torch.compile()` 완전 호환. PyG는 `torch_geometric.compile()` 래퍼를 제공하여 `MessagePassing` 레이어의 컴파일러 친화성을 높임.

### 벤치마크 (PyG 공식)

| 모델 | Speedup | 그래프 규모 |
|------|---------|------------|
| GCN | **2.83x** | 합성 10K 노드 |
| GraphSAGE | **2.24x** | 합성 10K 노드 |
| GIN | **~3x** | 합성 10K 노드 |

### 적용 방법

```python
import torch_geometric

# 기존 코드
encoder = GATEncoder(input_dim=384, hidden_dim=256, output_dim=384)

# 1줄 추가
encoder = torch_geometric.compile(encoder)
```

### 주의사항 (Graph Breaks)

| 원인 | 해결 방법 |
|------|-----------|
| `global_mean_pool()` | `batch` 인자 명시적 전달 |
| `remove_self_loops()` / `add_remaining_self_loops()` | 모델 입력 전 transform으로 처리, `add_self_loops=False` |
| 데이터 종속 제어 흐름 (if tensor condition) | 상수 조건으로 변환 |
| 동적 그래프 크기 | `dynamic=True` 설정 |

### 이 프로젝트 적용 시 고려사항
- `GATEncoder`, `GCNEncoder`: 문제 없이 컴파일 가능
- `GraphTransformerEncoder` (GPSConv): global attention 포함 → graph break 가능성 있음
- `KGEWrapper`: 내부 PyG `KGEModel`은 컴파일 미지원일 수 있음

**결론**: GNN 인코더에 `torch_geometric.compile()` 적용은 가장 높은 ROI 최적화.

---

## 6. Mixed Precision (bfloat16)

### GNN에서의 FP16 vs bfloat16

| 속성 | FP16 | bfloat16 |
|------|------|----------|
| 지수 비트 | 5 | 8 |
| 가수 비트 | 10 | 7 |
| Dynamic range | 좁음 | FP32 동일 |
| GradScaler 필요 | **필요** | **불필요** |
| Ampere GPU 지원 | 지원 | 지원 |

### GNN scatter 연산의 수치 안정성

GNN의 neighborhood aggregation (scatter_sum, scatter_mean)은 대규모 합산을 수행:
- **FP16**: 좁은 dynamic range로 overflow 위험 (특히 고차수 노드)
- **bfloat16**: FP32 동일한 range → overflow 없음
- **권장**: **bfloat16** 사용 (GradScaler 불필요, 수치 안정적)

### RTX 3070 (Ampere) 지원
- SM 8.6 아키텍처 → bfloat16 네이티브 지원
- TF32 tensor core도 활용 가능

### 적용 코드 (학습 루프)

```python
from torch.amp import autocast

# 학습 루프 내
with autocast(device_type="cuda", dtype=torch.bfloat16):
    z = encoder(x, edge_index)
    pos_scores = decoder(z[pos_ei[0]], z[pos_ei[1]])
    neg_scores = decoder(z[neg_ei[0]], z[neg_ei[1]])
    loss = F.binary_cross_entropy_with_logits(scores, labels)

# bfloat16은 GradScaler 불필요
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### 이 프로젝트에서의 효과
- GAT (1.8M params): ~7MB → ~3.5MB 모델 메모리 절약
- 14K 노드 특성 텐서: 13920 × 384 × 4B = 20MB → 10MB
- **총 GPU 메모리 절감**: 약 50% (실질적 이점은 모델이 클수록 증가)
- 현재 GAT 학습이 ~350MB로 충분히 작아 필수는 아님

---

## 7. 이 프로젝트 기준 종합 권고

### 현재 환경

```
PyTorch:        2.10.0+cu128
PyG:            2.7.0
GPU:            RTX 3070 (8GB, Ampere SM 8.6)
Graph scale:    14K nodes, 19K edges (LPG) / 12K nodes, 13K edges (RDF)
pyg-lib:        NOT installed
torch-scatter:  NOT installed
```

### 우선순위별 권고

#### 1순위: torch.compile() (코드 1줄, 높은 효과)
```python
encoder = torch_geometric.compile(encoder)
```
- 예상 효과: 2-3x 학습 가속
- 리스크: 낮음 (graph break 시 자동 eager fallback)

#### 2순위: pyg-lib / torch-scatter 설치 (설치만, 코드 변경 없음)
```bash
pip install pyg_lib torch_scatter -f https://data.pyg.org/whl/torch-2.10.0+cu128.html
```
- 예상 효과: scatter 연산 ~2x 가속
- 리스크: 휠 호환성 (PyTorch 2.10 대응 휠 미확인)

#### 3순위: bfloat16 AMP (코드 ~10줄)
- 예상 효과: 메모리 50% 절약, 약간의 속도 향상
- 리스크: 수렴 검증 필요

#### 불필요: cuGraph-PyG
- 14K 노드에서 mini-batch sampling 불필요 (full-graph training)
- cugraph-ops fused conv 제거됨
- conda 생태계 충돌 리스크

---

## 8. 학습 결과 요약 (2026-02-08)

### GNN Models (LPG, Link Prediction)

| Model | Params | Hidden | Heads | Test MRR | Hits@1 | Hits@10 | Early Stop |
|-------|--------|--------|-------|----------|--------|---------|------------|
| **GCN** | 329K | 256 | - | **0.0353** | 0.0132 | **0.0662** | Epoch 15 |
| GAT | 1.8M | 256 | 4 | 0.0240 | 0.0032 | 0.0503 | Epoch 15 |
| GraphTransformer | 126K | 64 | 1* | 0.0136 | 0.0037 | 0.0270 | Epoch 15 |

*GraphTransformer: hidden_dim=256/heads=4 설정은 O(n^2) attention으로 RTX 3070 OOM. hidden_dim=64/heads=1로 축소 실행.

### KGE Models (RDF, Link Prediction)

| Model | Params | Hidden | Test MRR | Hits@10 | Mean Rank | Early Stop |
|-------|--------|--------|----------|---------|-----------|------------|
| **DistMult** | 4.1M | 256 | **0.0773** | **0.1421** | 4874 | Epoch 35 |
| TransE | 4.1M | 256 | 0.0534 | 0.1095 | 5198 | Epoch 45 |
| RotatE | 3.7M | 128 | 0.0506 | 0.1071 | **3734** | Epoch 45 |
| ComplEx | 4.1M | 128 | 0.0493 | 0.1000 | 5146 | Epoch 20 |

### 분석
- 모든 모델이 일찍 early stopping → 15K/12K edge 규모에서 빠르게 수렴
- KGE가 GNN보다 높은 MRR → RDF 트리플의 직접적 관계 학습이 소규모 그래프에서 유리
- DistMult이 최고 MRR (0.0773) — 대칭 관계가 많은 금융 온톨로지에 적합
- RotatE가 최저 Mean Rank (3734) — 비대칭 관계 모델링에서 강점
- GraphTransformer는 GPU 메모리 제약으로 본래 성능 발휘 불가 → Colab A100에서 재실험 권장
