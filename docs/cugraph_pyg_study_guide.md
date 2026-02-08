# cuGraph-PyG 심층 학습 가이드

> 소스: [rapidsai/cugraph-gnn](https://github.com/rapidsai/cugraph-gnn/tree/main/python/cugraph-pyg) v26.04
> 작성일: 2026-02-08

---

## 목차

1. [왜 cuGraph-PyG인가](#1-왜-cugraph-pyg인가)
2. [전체 아키텍처](#2-전체-아키텍처)
3. [PyG의 Remote Backend 패턴 이해](#3-pyg의-remote-backend-패턴-이해)
4. [Data Layer: GraphStore & FeatureStore](#4-data-layer-graphstore--featurestore)
5. [Sampler Layer: GPU 네이버 샘플링](#5-sampler-layer-gpu-네이버-샘플링)
6. [Loader Layer: NeighborLoader 계열](#6-loader-layer-neighborloader-계열)
7. [Tensor Layer: WholeGraph 분산 텐서](#7-tensor-layer-wholegraph-분산-텐서)
8. [전체 워크플로우: 바닐라 PyG vs cuGraph-PyG](#8-전체-워크플로우-바닐라-pyg-vs-cugraph-pyg)
9. [cugraph-ops의 탄생과 죽음](#9-cugraph-ops의-탄생과-죽음)
10. [멀티 GPU 학습 패턴](#10-멀티-gpu-학습-패턴)
11. [규모별 기술 선택 가이드](#11-규모별-기술-선택-가이드)
12. [부록: 의존성과 설치](#부록-의존성과-설치)

---

## 1. 왜 cuGraph-PyG인가

### GNN 학습의 병목

대규모 그래프에서 GNN 학습의 실제 병목은 **GNN 연산 자체가 아니다**:

```
전체 학습 시간 분해 (billion-edge 그래프):

  ┌──────────────────────────────────────────────────┐
  │ Neighbor Sampling        ████████████████  60%    │  ← CPU 병목
  │ Feature Fetching         ████████          30%    │  ← PCIe/메모리 병목
  │ GNN Forward/Backward     ███               10%    │  ← GPU 연산
  └──────────────────────────────────────────────────┘
```

PyG의 기본 `NeighborLoader`는 CPU에서 샘플링 → PCIe로 GPU 전송. 이 파이프라인에서:
- **Neighbor sampling**: CPU 싱글스레드, 랜덤 메모리 접근 → 느림
- **Feature lookup**: 샘플된 노드의 특성을 CPU 메모리에서 읽어 GPU로 복사 → PCIe 대역폭 한계
- **GNN 연산**: GPU에서 수행 → 대부분의 시간을 데이터 대기로 소비

cuGraph-PyG는 이 **sampling + feature fetching**을 GPU로 옮긴다.

### cuGraph-PyG의 위치

```
┌─────────────────────────────────────────────────────────┐
│                   GNN 학습 파이프라인                      │
│                                                          │
│   ┌─────────┐    ┌──────────┐    ┌──────────┐           │
│   │  Graph   │───→│ Neighbor │───→│  Feature │───→ GNN   │
│   │ Storage  │    │ Sampling │    │ Fetching │    연산    │
│   └─────────┘    └──────────┘    └──────────┘           │
│                                                          │
│   바닐라 PyG:    CPU           CPU           GPU          │
│   cuGraph-PyG:   GPU           GPU           GPU ← 전부! │
└─────────────────────────────────────────────────────────┘
```

---

## 2. 전체 아키텍처

### 패키지 구조

```
cugraph_pyg/
├── data/                          # ← PyG의 Remote Backend 구현
│   ├── graph_store.py             #    GraphStore (그래프 토폴로지)
│   └── feature_store.py           #    FeatureStore (노드/엣지 특성)
│
├── sampler/                       # ← GPU 네이버 샘플링 엔진
│   ├── sampler.py                 #    BaseSampler (PyG 호환 래퍼)
│   ├── distributed_sampler.py     #    DistributedNeighborSampler (핵심 엔진)
│   ├── sampler_utils.py           #    유틸리티 (negative sampling 등)
│   └── io.py                      #    BufferedSampleReader
│
├── loader/                        # ← PyG Loader의 Duck-type 구현
│   ├── node_loader.py             #    NodeLoader
│   ├── neighbor_loader.py         #    NeighborLoader (GraphSAGE-style)
│   ├── link_loader.py             #    LinkLoader
│   └── link_neighbor_loader.py    #    LinkNeighborLoader (링크 예측용)
│
├── tensor/                        # ← WholeGraph 분산 텐서
│   ├── dist_tensor.py             #    DistTensor
│   ├── dist_matrix.py             #    DistMatrix
│   └── utils.py                   #    NVLink 감지, 메모리 타입 선택
│
└── examples/                      # ← 실전 예제
    ├── gcn_dist_mnmg.py           #    GCN 노드분류 (멀티GPU)
    ├── rgcn_link_class_mnmg.py    #    R-GCN 링크분류 (멀티GPU)
    ├── dist_gin_sg.py             #    GIN 그래프분류 (싱글GPU)
    ├── mag_lp_mnmg.py             #    MAG 링크예측
    ├── movielens_mnmg.py          #    MovieLens 추천
    └── kg/                        #    지식그래프 예제
```

### 레이어 간 관계

```
사용자 코드
    │
    ▼
NeighborLoader (loader/)
    │  내부적으로 생성
    ▼
BaseSampler (sampler/)
    │  래핑
    ▼
DistributedNeighborSampler (sampler/)
    │  pylibcugraph C++ 호출
    ▼
GraphStore + FeatureStore (data/)
    │  WholeGraph 백엔드
    ▼
pylibcugraph  ←  CUDA C++ 샘플링
pylibwholegraph  ←  분산 메모리 관리
```

**핵심**: 사용자는 `NeighborLoader`만 만들면 된다. 나머지는 내부적으로 자동 구성.

---

## 3. PyG의 Remote Backend 패턴 이해

cuGraph-PyG를 이해하려면 먼저 PyG 2.4+의 **Remote Backend** 패턴을 알아야 한다.

### 기존 PyG: 모든 것이 메모리에

```python
# 전통적인 PyG — 모든 데이터가 CPU/GPU 텐서
data = Data(
    x=torch.randn(100000, 128),           # 노드 특성: CPU 텐서
    edge_index=torch.randint(0, 100000, (2, 500000)),  # 엣지: CPU 텐서
)

# 전부 메모리에 올려야 함
loader = NeighborLoader(data, num_neighbors=[25, 10], batch_size=512)
```

문제: 10억 노드 그래프의 특성을 메모리에 올릴 수 없다.

### Remote Backend: 인터페이스 분리

PyG 2.4+는 데이터를 **어디에 저장할지**와 **어떻게 접근할지**를 분리:

```python
# PyG Remote Backend 인터페이스
class GraphStore(ABC):
    """그래프 토폴로지 저장 (엣지)"""
    def _put_edge_index(self, edge_index, edge_attr): ...
    def _get_edge_index(self, edge_attr): ...
    def _remove_edge_index(self, edge_attr): ...

class FeatureStore(ABC):
    """노드/엣지 특성 저장"""
    def _put_tensor(self, tensor, attr): ...
    def _get_tensor(self, attr): ...
    def _remove_tensor(self, attr): ...
```

이 인터페이스를 구현하면 **데이터가 어디에 있든** PyG의 Loader가 작동:

| 구현체 | 데이터 위치 | 용도 |
|--------|-----------|------|
| PyG 기본 `Data` | CPU/GPU 메모리 | 소규모 그래프 |
| **cuGraph-PyG** | **GPU 메모리 (cuGraph)** | **대규모 GPU 학습** |
| 가상의 Redis 구현 | 원격 Redis 서버 | 분산 저장 |

### cuGraph-PyG는 이 인터페이스의 GPU 구현

```python
# cuGraph-PyG의 구현
from cugraph_pyg.data import GraphStore, FeatureStore

# GPU에 그래프 저장
graph_store = GraphStore()                 # pylibcugraph.SGGraph 백엔드
feature_store = FeatureStore()             # WholeGraph WholeMemory 백엔드

# PyG의 Loader에 (feature_store, graph_store) 튜플로 전달
loader = NeighborLoader(
    (feature_store, graph_store),          # ← Remote Backend 튜플!
    num_neighbors=[25, 10],
    ...
)
```

---

## 4. Data Layer: GraphStore & FeatureStore

### 4.1 GraphStore

GPU에 그래프 토폴로지를 저장. 내부적으로 `pylibcugraph`의 C++ 그래프 자료구조 사용.

```python
class GraphStore(torch_geometric.data.GraphStore):
    # 조건부 상속: PyG가 없으면 object에서 상속
```

#### 엣지 저장

```python
graph_store = GraphStore()

# 동질 그래프 (homogeneous)
graph_store[("node", "edge", "node")] = edge_index  # [2, num_edges] COO

# 이종 그래프 (heterogeneous)
graph_store[("user", "buys", "item")] = user_buys_item_edges
graph_store[("user", "rates", "movie")] = user_rates_movie_edges
```

#### 내부 그래프 생성 (Lazy)

`_graph` 프로퍼티가 처음 접근될 때 `pylibcugraph` 그래프를 생성:

```python
@property
def _graph(self):
    # 싱글 GPU → pylibcugraph.SGGraph
    # 멀티 GPU → pylibcugraph.MGGraph
    # CSR 형식으로 변환하여 저장 (샘플링 최적화)
```

#### 주요 프로퍼티

```python
graph_store.is_multi_gpu       # bool: 멀티 GPU 모드인가
graph_store.is_homogeneous     # bool: 동질 그래프인가
graph_store._vertex_offsets    # Dict: 정점 타입별 글로벌 오프셋
graph_store._numeric_edge_types  # 엣지 타입의 숫자 인코딩
```

### 4.2 FeatureStore

노드/엣지 특성을 **WholeGraph WholeMemory**에 저장. 멀티 GPU에서 데이터 복제 없이 접근 가능.

```python
class FeatureStore(torch_geometric.data.FeatureStore):
    def __init__(self, memory_type=None, location="cpu"):
        # memory_type: None이면 자동 감지
        #   - NVLink 있으면: "distributed" (GPU 메모리 분산)
        #   - NVLink 없으면: "chunked" (CPU 메모리 + GPU 캐시)
        # location: "cpu" 또는 "cuda"
```

#### 특성 저장

```python
feature_store = FeatureStore()

# 동질 그래프
feature_store[("node", "x", None)] = node_features       # [N, feature_dim]
feature_store[("node", "y", None)] = node_labels          # [N]

# 이종 그래프
feature_store[("user", "x", None)] = user_features
feature_store[("item", "x", None)] = item_features

# 엣지 특성
feature_store[("user", "buys", "item"), "edge_attr", None] = edge_weights
```

#### 내부: DistTensor / DistEmbedding

```python
# FeatureStore._put_tensor 내부:
if tensor.is_floating_point():
    self._tensors[key] = DistTensor(tensor)       # 연속 특성
else:
    self._tensors[key] = DistEmbedding(tensor)    # 이산 특성/레이블
```

- `DistTensor`: WholeMemory 백엔드의 분산 텐서. GPU 간 zero-copy 접근.
- `DistEmbedding`: 임베딩 테이블. 학습 가능한 파라미터도 분산 저장 가능.

### 4.3 NVLink 감지와 메모리 전략

```python
# cugraph_pyg/tensor/utils.py
def has_nvlink_network():
    """NVLink 토폴로지 감지 → 메모리 전략 결정"""
    # NVLink 있음 → "distributed": 각 GPU가 데이터 일부를 소유, 직접 접근
    # NVLink 없음 → "chunked": CPU 메모리에 저장, GPU 캐시로 접근
```

이것이 중요한 이유:
- **NVLink** (A100/H100 서버): GPU 간 600GB/s 대역폭 → 원격 GPU 메모리 직접 읽기 가능
- **PCIe only** (일반 서버/RTX): GPU 간 32GB/s → CPU 메모리 경유가 더 효율적

---

## 5. Sampler Layer: GPU 네이버 샘플링

### 5.1 샘플링이 왜 병목인가

GraphSAGE-style 학습에서 각 배치마다:

```
Target node → 1-hop 이웃 25개 샘플 → 2-hop 이웃 10개씩 샘플
= 1 + 25 + 250 = 276 노드의 서브그래프

이것을 매 배치(512개 타겟)마다 반복:
= 512 × 276 ≈ 141,000 노드 접근/배치
```

CPU에서 이 랜덤 접근은 **캐시 미스의 연속**. cuGraph는 이를 GPU의 대규모 병렬 처리로 해결.

### 5.2 DistributedNeighborSampler

핵심 샘플링 엔진. `pylibcugraph` C++ 함수를 직접 호출.

```python
class DistributedNeighborSampler(BaseDistributedSampler):
    # GPU 메모리 기반 자동 배치 크기 튜닝
    BASE_VERTICES_PER_BYTE = 0.1107662486009992

    def __init__(
        self,
        graph,                           # pylibcugraph.SGGraph 또는 MGGraph
        *,
        fanout=[-1],                     # 레이어별 이웃 수 (-1 = 전체)
        compression="COO",               # 출력 형식: "COO" 또는 "CSC"
        biased=False,                    # 가중치 기반 샘플링
        heterogeneous=False,             # 이종 그래프
        temporal=False,                  # 시간 기반 샘플링
        prior_sources_behavior="exclude", # 이전 홉 소스 처리
        deduplicate_sources=True,        # 소스 노드 중복 제거
        local_seeds_per_call=None,       # None → GPU 메모리 기반 자동 계산
    ):
```

#### 8가지 샘플링 구성

`_func_table`이 `(homogeneous, biased, temporal)` 조합을 전용 C++ 함수에 매핑:

```python
# 내부 함수 테이블 (개념적 표현)
_func_table = {
    (homo=True,  biased=False, temporal=False): pylibcugraph.uniform_neighbor_sample,
    (homo=True,  biased=True,  temporal=False): pylibcugraph.biased_neighbor_sample,
    (homo=True,  biased=False, temporal=True):  pylibcugraph.temporal_neighbor_sample,
    (homo=True,  biased=True,  temporal=True):  pylibcugraph.temporal_biased_sample,
    (homo=False, biased=False, temporal=False): pylibcugraph.heterogeneous_uniform_sample,
    (homo=False, biased=True,  temporal=False): pylibcugraph.heterogeneous_biased_sample,
    (homo=False, biased=False, temporal=True):  pylibcugraph.heterogeneous_temporal_sample,
    (homo=False, biased=True,  temporal=True):  pylibcugraph.heterogeneous_temporal_biased,
}
```

#### GPU 메모리 기반 자동 튜닝

```python
# local_seeds_per_call이 None이면 자동 계산:
if local_seeds_per_call is None:
    available_bytes = torch.cuda.get_device_properties(0).total_memory * 0.5
    local_seeds_per_call = int(available_bytes * BASE_VERTICES_PER_BYTE)
    # RTX 3070 (8GB): ~0.5 * 8GB * 0.11 ≈ 450K seeds/call
    # A100 (80GB):    ~0.5 * 80GB * 0.11 ≈ 4.5M seeds/call
```

### 5.3 BaseSampler (PyG 호환 래퍼)

`DistributedNeighborSampler`의 출력을 PyG의 `SamplerOutput`/`HeteroSamplerOutput` 형식으로 변환:

```python
class BaseSampler:
    def __init__(self, sampler, data, batch_size=16):
        self._sampler = sampler           # DistributedNeighborSampler
        self._data = data                 # (FeatureStore, GraphStore) 튜플

    def sample_from_nodes(self, index) -> Iterator[SamplerOutput]:
        """노드 기반 샘플링 → PyG SamplerOutput"""
        raw = self._sampler.sample_from_nodes(index.node, batch_size=...)
        # SampleReader로 디코딩 → SamplerOutput으로 변환
        yield from SampleIterator(self._data, decoded_outputs)

    def sample_from_edges(self, index, neg_sampling=None) -> Iterator[SamplerOutput]:
        """엣지 기반 샘플링 → PyG SamplerOutput (링크 예측용)"""
        ...
```

### 5.4 SampleReader 계층

C++ 샘플러의 raw 출력을 PyG 형식으로 디코딩:

```python
class HomogeneousSampleReader(SampleReader):
    """동질 그래프: COO/CSC 디코딩"""
    def _decode(self, raw_sample_data, index):
        # raw C++ 출력 → (edge_index, node_ids, batch) 변환

class HeterogeneousSampleReader(SampleReader):
    """이종 그래프: 타입별 분할 + COO 디코딩"""
    def __init__(self, base_reader, src_types, dst_types,
                 vertex_offsets, edge_types, vertex_types):
        # 정점/엣지 타입 오프셋으로 이종 그래프 복원
```

---

## 6. Loader Layer: NeighborLoader 계열

### 6.1 NeighborLoader

가장 많이 사용하는 클래스. GraphSAGE-style 미니배치 학습용.

```python
class NeighborLoader(NodeLoader):
    def __init__(
        self,
        data,                    # Data | HeteroData | (FeatureStore, GraphStore)
        num_neighbors,           # [25, 10] 또는 {edge_type: [25, 10]}
        input_nodes=None,        # 학습 대상 노드 (None = 전체)
        batch_size=16,           # 미니배치 크기
        replace=False,           # 복원 추출 여부
        subgraph_type="directional",  # 서브그래프 방향
        disjoint=False,          # 배치 내 서브그래프 분리
        temporal_strategy="uniform",  # 시간 샘플링 전략
        time_attr=None,          # 시간 속성 이름
        weight_attr=None,        # 엣지 가중치 속성
        compression=None,        # "COO" 또는 None(자동)
        local_seeds_per_call=None,  # GPU 메모리 기반 자동 튜닝
        shuffle=False,
        drop_last=False,
        ...
    ):
```

#### 내부 동작 흐름

```
NeighborLoader.__init__()
    │
    ├── GraphStore에서 그래프 정보 추출
    ├── DistributedNeighborSampler 생성
    │     └── fanout = num_neighbors
    │     └── compression 결정
    │     └── biased = (weight_attr is not None)
    └── BaseSampler로 래핑

NeighborLoader.__iter__()
    │
    ├── input_nodes를 batch_size 단위로 분할
    ├── 각 배치에 대해:
    │     ├── BaseSampler.sample_from_nodes() 호출
    │     │     └── DistributedNeighborSampler.sample_from_nodes()
    │     │           └── pylibcugraph C++ 샘플링
    │     ├── SampleReader로 디코딩
    │     └── SampleIterator가 FeatureStore에서 특성 조회
    └── yield PyG Data/HeteroData (서브그래프 + 특성)
```

### 6.2 LinkNeighborLoader

링크 예측 전용. 엣지 배치를 샘플링.

```python
class LinkNeighborLoader(LinkLoader):
    def __init__(
        self,
        data,
        num_neighbors,
        edge_label_index,        # 학습 대상 엣지
        edge_label=None,         # 엣지 레이블 (분류용)
        edge_label_time=None,    # 엣지 시간 (temporal용)
        neg_sampling=None,       # NegativeSampling 객체
        neg_sampling_ratio=None, # negative 비율
        batch_size=16,
        ...
    ):
```

**제약사항** (vs NeighborLoader):
- `subgraph_type`은 `"directional"`만 지원
- `disjoint=True` 불가

### 6.3 Duck Typing 패턴

cuGraph-PyG의 Loader는 **PyG 클래스를 상속하지 않는다**:

```python
# PyG 코드:
class NeighborLoader(NodeLoader):  # torch_geometric.loader.NodeLoader 상속
    ...

# cuGraph-PyG 코드:
class NeighborLoader(NodeLoader):  # cugraph_pyg.loader.NodeLoader (자체 정의)
    ...
```

왜 Duck typing인가:
- PyG를 하드 의존성으로 만들지 않음 (`import_optional` 사용)
- PyG 버전 변경에 유연하게 대응
- 인터페이스만 동일하면 사용자 코드 변경 불필요

```python
# 사용자 관점에서는 동일:

# 바닐라 PyG
from torch_geometric.loader import NeighborLoader

# cuGraph-PyG (drop-in 교체)
from cugraph_pyg.loader import NeighborLoader

# 이후 코드 동일
loader = NeighborLoader(data, num_neighbors=[25, 10], batch_size=512)
for batch in loader:
    out = model(batch.x, batch.edge_index)
```

---

## 7. Tensor Layer: WholeGraph 분산 텐서

### 7.1 WholeGraph란

NVIDIA의 분산 그래프 저장 라이브러리. 핵심 아이디어: **데이터를 복제하지 않고 여러 GPU에 분산 저장**.

```
기존 DataParallel:
  GPU 0: [전체 특성 복사본]    ← 메모리 낭비
  GPU 1: [전체 특성 복사본]
  GPU 2: [전체 특성 복사본]
  GPU 3: [전체 특성 복사본]

WholeGraph:
  GPU 0: [특성 0~25%]         ← 각 GPU가 1/4만 저장
  GPU 1: [특성 25~50%]            NVLink로 다른 GPU 데이터 직접 접근
  GPU 2: [특성 50~75%]
  GPU 3: [특성 75~100%]

  메모리 사용: 1/4 (4 GPU 기준)
```

### 7.2 DistTensor

```python
class DistTensor:
    """WholeMemory 백엔드의 분산 텐서"""

    def __init__(self, tensor, memory_type=None, location=None):
        # tensor: 초기 데이터 (로컬 PyTorch 텐서)
        # memory_type: "distributed" | "chunked" | "continuous"
        # location: "cpu" | "cuda"
        ...

    def __getitem__(self, index):
        """인덱스로 접근 — WholeMemory가 데이터 위치 자동 해결"""
        # index가 로컬 GPU에 있으면 → 직접 읽기
        # index가 원격 GPU에 있으면 → NVLink/PCIe로 읽기
        ...
```

### 7.3 DistEmbedding

```python
class DistEmbedding:
    """분산 임베딩 테이블 — 학습 가능"""

    def __init__(self, num_embeddings, embedding_dim, ...):
        # WholeMemory에 임베딩 테이블 분산 저장
        # 각 GPU가 일부 임베딩만 소유
        ...

    def forward(self, index):
        """임베딩 lookup — 자동으로 올바른 GPU에서 가져옴"""
        ...
```

### 7.4 메모리 타입 비교

| 타입 | 데이터 위치 | 접근 방식 | 최적 환경 |
|------|-----------|-----------|-----------|
| `"distributed"` | GPU 메모리 분산 | NVLink 직접 접근 | NVLink 서버 (DGX) |
| `"chunked"` | CPU 메모리 | GPU 캐시 + CPU fallback | PCIe only 서버 |
| `"continuous"` | 단일 연속 메모리 | 직접 접근 | 싱글 GPU |

---

## 8. 전체 워크플로우: 바닐라 PyG vs cuGraph-PyG

### 8.1 노드 분류 비교

#### 바닐라 PyG

```python
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GCNConv

# 1. 데이터 로드 (CPU 메모리)
dataset = Planetoid(root='data', name='Cora')
data = dataset[0]

# 2. NeighborLoader (CPU 샘플링)
loader = NeighborLoader(
    data,
    num_neighbors=[25, 10],
    batch_size=512,
    input_nodes=data.train_mask,
)

# 3. 학습
model = GCN(dataset.num_features, 256, dataset.num_classes)
for batch in loader:
    batch = batch.to('cuda')  # ← CPU→GPU 전송 필요
    out = model(batch.x, batch.edge_index)
    loss = F.cross_entropy(out[:batch.batch_size], batch.y[:batch.batch_size])
    loss.backward()
    optimizer.step()
```

#### cuGraph-PyG

```python
import torch
import cugraph_pyg
from cugraph_pyg.data import GraphStore, FeatureStore
from cugraph_pyg.loader import NeighborLoader
from torch_geometric.nn import GCNConv  # ← GNN 레이어는 동일!

# 1. GPU 스토어 생성
feature_store = FeatureStore()
graph_store = GraphStore()

# 2. 데이터를 GPU 스토어에 로드
feature_store[("node", "x", None)] = node_features
feature_store[("node", "y", None)] = node_labels
graph_store[("node", "rel", "node")] = edge_index

# 3. NeighborLoader (GPU 샘플링)
loader = NeighborLoader(
    (feature_store, graph_store),  # ← Remote Backend 튜플
    num_neighbors=[25, 10],
    batch_size=512,
    input_nodes=train_nodes,
)

# 4. 학습 (데이터가 이미 GPU에 있음)
model = GCN(num_features, 256, num_classes).to('cuda')
for batch in loader:
    # batch는 이미 GPU 텐서 — 전송 불필요!
    out = model(batch.x, batch.edge_index)
    loss = F.cross_entropy(out[:batch.batch_size], batch.y[:batch.batch_size])
    loss.backward()
    optimizer.step()
```

**차이점 요약**:

| 항목 | 바닐라 PyG | cuGraph-PyG |
|------|-----------|-------------|
| 데이터 저장 | CPU 텐서 (`Data`) | GPU 스토어 (`GraphStore` + `FeatureStore`) |
| 샘플링 위치 | CPU | GPU (pylibcugraph C++) |
| 배치 전송 | `.to('cuda')` 필요 | 이미 GPU에 있음 |
| GNN 레이어 | `torch_geometric.nn.*` | **동일** |
| import 변경 | — | `NeighborLoader`만 교체 |

### 8.2 링크 예측 (R-GCN, 이종 그래프)

`examples/rgcn_link_class_mnmg.py` 기반:

```python
import cugraph_pyg
from torch_geometric.nn import FastRGCNConv

# 1. 스토어 생성
feature_store = FeatureStore()
graph_store = GraphStore()

# 2. 이종 그래프 데이터 로드
# 학습 가능한 노드 임베딩 (Xavier 초기화)
node_emb = torch.nn.init.xavier_uniform_(
    torch.empty(num_nodes, hidden_channels)
)
feature_store[("n", "emb", None)] = node_emb
feature_store[("n", "e", "n"), "edge_type", None] = edge_types.int()
graph_store[("n", "e", "n")] = edge_index

# 3. LinkNeighborLoader (엣지 배치 + 네거티브 샘플링)
from cugraph_pyg.loader import LinkNeighborLoader

loader = LinkNeighborLoader(
    (feature_store, graph_store),
    num_neighbors=[fan_out] * num_layers,
    edge_label_index=train_edges,
    batch_size=16384,
    shuffle=True,
    # neg_sampling 설정 가능
)

# 4. R-GCN 학습
class RGCNEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = FastRGCNConv(hidden_channels, hidden_channels, num_relations)
        self.conv2 = FastRGCNConv(hidden_channels, hidden_channels, num_relations)

    def forward(self, edge_index, edge_type):
        x = self.emb.weight  # 학습 가능한 임베딩
        x = self.conv1(x, edge_index, edge_type).relu()
        x = self.conv2(x, edge_index, edge_type)
        return x

model = RGCNEncoder().to('cuda')
for batch in loader:
    z = model(batch.edge_index, batch.edge_type)
    loss = model.recon_loss(z, batch.edge_label_index)
    loss.backward()
    optimizer.step()
```

### 8.3 우리 프로젝트와의 대응

```
우리 프로젝트 (14K 노드, full-graph training):
  ┌──────────────────────────────────┐
  │ Neo4j → LPGGraphBuilder → Data   │  ← CPU 메모리에 전체 그래프
  │ Data.to('cuda')                   │  ← 한번에 GPU로 전송 (23MB)
  │ z = encoder(x, edge_index)        │  ← 전체 노드 인코딩
  │ loss on sampled edges             │
  └──────────────────────────────────┘
  샘플링 없음 → cuGraph 이점 없음

cuGraph-PyG가 빛나는 시나리오 (100M+ 노드):
  ┌──────────────────────────────────┐
  │ GraphStore (GPU)                  │  ← 그래프가 GPU 메모리에
  │ NeighborLoader(batch_size=512)    │  ← GPU에서 25+10 이웃 샘플링
  │ for batch in loader:              │  ← 미니배치 학습
  │     z = encoder(batch.x, ...)     │  ← 서브그래프만 연산
  └──────────────────────────────────┘
  CPU 샘플링이 병목 → cuGraph가 10x+ 가속
```

---

## 9. cugraph-ops의 탄생과 죽음

### 있었던 것: Fused GNN Convolutions

RAPIDS 24.x까지 `cugraph-ops`는 GPU-fused GNN 레이어를 제공:

```python
# 과거 코드 (더 이상 작동하지 않음)
from torch_geometric.nn.conv.cugraph import CuGraphGATConv, CuGraphSAGEConv

# PyG의 GATConv 대신 drop-in 사용
conv = CuGraphGATConv(in_channels, out_channels, heads=4)
# → 내부적으로 cugraph-ops의 fused CUDA 커널 사용
# → message passing + aggregation이 하나의 커널로 실행
```

### 제거된 이유

[RSN0041](https://docs.rapids.ai/notices/rsn0041/) — RAPIDS 25.02에서 제거:

1. **유지보수 부담**: 각 GNN 레이어마다 별도 CUDA 커널 필요
2. **PyG 업데이트 추적 어려움**: PyG 버전 올라갈 때마다 fused 커널도 수정 필요
3. **torch.compile()의 등장**: PyG 2.5+에서 `torch.compile()`이 비슷한 fusion 효과 달성
4. **전략 전환**: NVIDIA가 "데이터 파이프라인 가속"에 집중하기로 결정

### 제거된 컴포넌트

```
삭제됨:
  - CuGraphGATConv
  - CuGraphSAGEConv
  - CuGraphRGCNConv
  - libcugraphops (C++ 라이브러리)
  - pylibcugraphops (Python 바인딩)

유지됨:
  - cugraph_pyg.data.GraphStore
  - cugraph_pyg.data.FeatureStore
  - cugraph_pyg.loader.NeighborLoader
  - cugraph_pyg.sampler.*
```

### 현재 권장 대안

```python
# 레이어 가속은 torch.compile()로:
import torch_geometric
from torch_geometric.nn import GATConv

model = GAT(...)
model = torch_geometric.compile(model)  # ← torch.compile()이 fusion 담당

# 데이터 파이프라인 가속은 cuGraph-PyG로:
from cugraph_pyg.loader import NeighborLoader  # ← 샘플링만 담당
```

---

## 10. 멀티 GPU 학습 패턴

### 실행 방법

```bash
# torchrun으로 NCCL 기반 분산 실행
torchrun \
    --nproc_per_node=4 \           # GPU 4개
    --nnodes=1 \                    # 노드 1개
    examples/gcn_dist_mnmg.py \
    --dataset ogbn-papers100M \
    --fan_out 25,10 \
    --batch_size 512
```

### 내부 통신 구조

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   GPU 0     │  │   GPU 1     │  │   GPU 2     │  │   GPU 3     │
│             │  │             │  │             │  │             │
│ GraphStore  │  │ GraphStore  │  │ GraphStore  │  │ GraphStore  │
│ (MGGraph)   │══│ (MGGraph)   │══│ (MGGraph)   │══│ (MGGraph)   │
│             │  │             │  │             │  │             │
│ FeatureStore│  │ FeatureStore│  │ FeatureStore│  │ FeatureStore│
│ (WholeGraph)│══│ (WholeGraph)│══│ (WholeGraph)│══│ (WholeGraph)│
│             │  │             │  │             │  │             │
│ Sampler     │  │ Sampler     │  │ Sampler     │  │ Sampler     │
│ (독립)      │  │ (독립)      │  │ (독립)      │  │ (독립)      │
│             │  │             │  │             │  │             │
│ GNN Model   │  │ GNN Model   │  │ GNN Model   │  │ GNN Model   │
│ (DDP)       │══│ (DDP)       │══│ (DDP)       │══│ (DDP)       │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
       ║                ║                ║                ║
       ╚════════════════╩════════════════╩════════════════╝
                     NCCL / NVLink 통신
```

- **GraphStore**: `pylibcugraph.MGGraph`로 그래프 분산 저장
- **FeatureStore**: WholeGraph로 특성 분산 저장 (복제 없음)
- **Sampler**: 각 GPU가 독립적으로 자기 배치 샘플링
- **GNN Model**: PyTorch DDP로 gradient 동기화

### 분산 초기화 코드

```python
import torch.distributed as dist
from cugraph.gnn import cugraph_comms_init

# 1. PyTorch 분산 초기화
dist.init_process_group(backend="nccl")

# 2. cuGraph 통신 초기화 (RAFT 핸들)
cugraph_comms_init(
    rank=dist.get_rank(),
    world_size=dist.get_world_size(),
)

# 3. WholeGraph 초기화
import pylibwholegraph.torch as wgth
wgth.init(dist.get_rank(), dist.get_world_size())

# 이후 단일 GPU 코드와 거의 동일하게 작성
```

---

## 11. 규모별 기술 선택 가이드

### 의사결정 플로우차트

```
그래프 크기는?
│
├── < 100K 노드 (소규모)
│   └── Full-graph training
│       ├── torch.compile() 적용 (2-3x)
│       └── cuGraph 불필요
│
├── 100K ~ 10M 노드 (중규모)
│   └── Mini-batch training 필요
│       ├── PyG NeighborLoader (CPU) — 보통 충분
│       ├── pyg-lib 설치 — 샘플링 가속
│       └── cuGraph — 데이터 로딩이 병목이면
│
├── 10M ~ 1B 노드 (대규모)
│   └── cuGraph-PyG 권장
│       ├── GPU 샘플링 필수
│       ├── WholeGraph 분산 특성 저장
│       └── 멀티 GPU 학습
│
└── > 1B 노드 (초대규모)
    └── cuGraph-PyG + 멀티노드
        ├── torchrun --nnodes=N
        ├── WholeGraph 분산 저장
        └── MGGraph 분산 그래프
```

### 규모별 비교표

| 규모 | 예시 | Full-graph? | 기본 PyG | pyg-lib | cuGraph-PyG |
|------|------|-------------|----------|---------|-------------|
| 2K 노드 | Cora | O | 최적 | 미미한 차이 | 과잉 |
| **14K 노드** | **이 프로젝트** | **O** | **최적** | **약간 이점** | **불필요** |
| 170K 노드 | ogbn-arxiv | △ | 충분 | 권장 | 선택적 |
| 2.4M 노드 | ogbn-products | X | 가능 | 권장 | 권장 |
| 111M 노드 | ogbn-papers100M | X | 느림 | 필요 | **필수** |
| 1B+ 노드 | 산업 추천/소셜 | X | 불가 | 불충분 | **필수** |

### 이 프로젝트에 대한 결론

```
이 프로젝트 상태:
  - 그래프 크기: 14K 노드, 19K 엣지 (LPG) / 12K 노드, 13K 엣지 (RDF)
  - 학습 방식: Full-graph training (전체 노드 인코딩)
  - GPU 메모리: 그래프 데이터 23MB (8GB 중 0.3%)
  - 샘플링: 없음 (edge-level에서만 positive/negative 샘플링)

cuGraph-PyG가 불필요한 이유:
  1. Full-graph training → 네이버 샘플링 자체가 없음
  2. 전체 그래프가 GPU 메모리에 여유롭게 들어감
  3. 데이터 로딩이 병목이 아님 (1초 이내 완료)
  4. Python 3.10 환경 (cuGraph-PyG는 Python >= 3.11 요구)
  5. cugraph-ops (fused conv) 제거됨 → GNN 레이어 가속 효과 없음

실질적으로 유효한 최적화:
  1. torch.compile() → GNN 레이어 2-3x 가속 (1줄)
  2. pyg-lib → scatter 연산 가속 (설치만)
  3. bfloat16 AMP → 메모리 절약 (~10줄)
```

---

## 부록: 의존성과 설치

### cuGraph-PyG 설치 (참고용)

```bash
# conda 환경 (RAPIDS 공식 방법)
conda install -c rapidsai -c conda-forge -c nvidia \
    cugraph-pyg cuda-version=12.8

# pip (CUDA 12.x)
pip install cugraph-pyg-cu12
```

### 의존성 호환 매트릭스

| 패키지 | 버전 제약 | 이 프로젝트 | 호환 여부 |
|--------|-----------|-----------|-----------|
| Python | >= 3.11 | 3.10 | **X** |
| torch-geometric | >= 2.5, < 2.8 | 2.7.0 | O |
| torch | >= 2.9.0 | 2.10.0 | O |
| cupy-cuda13x | >= 13.6.0 | 미설치 | — |
| pylibcugraph | == 26.4.* | 미설치 | — |
| pylibwholegraph | == 26.4.* | 미설치 | — |

> Python 3.10이 첫 번째 blocker. cuGraph-PyG 도입 시 Python 3.11+ 마이그레이션 필요.

### 관련 저장소

| 저장소 | 역할 |
|--------|------|
| [rapidsai/cugraph-gnn](https://github.com/rapidsai/cugraph-gnn) | cuGraph-PyG + WholeGraph (메인) |
| [rapidsai/cugraph](https://github.com/rapidsai/cugraph) | cuGraph 코어 (graph analytics) |
| [pyg-team/pyg-lib](https://github.com/pyg-team/pyg-lib) | PyG low-level 최적화 |
| [pyg-team/pytorch_geometric](https://github.com/pyg-team/pytorch_geometric) | PyTorch Geometric 메인 |
