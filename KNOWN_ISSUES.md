# Known Issues & Improvement Backlog

> 프로젝트에서 발견된 문제와 개선 과제를 우선순위별로 관리한다.
> 완료된 항목은 [Resolved](#resolved) 섹션으로 이동.
>
> 관련 문서: [종합 보고서](docs/analysis/comprehensive_experiment_report.md) § 10. 개선 제안

---

## Critical — 실험 결과에 직접 영향

### RDF URI Prefix 토큰 낭비
- [ ] **RDF context의 ~57.8% 토큰이 FIBO URI prefix로 소비됨**
- 원인: `fibo-fnd-acc-4217:`, `fibo-fnd-rel-rel:` 등이 트리플마다 반복 (11 tokens/회)
- 해결: `src/utils/formatting.py`의 `format_rdf_cleaned()`를 실험 코드에 적용
- 예상 효과: RDF 토큰 ~40% 절감, attention density 10.6% → ~18%
- 관련 파일: `src/kvcache_experiment.py` `build_context()`, `src/attention_experiment.py`

### LPG "None" 노드 토큰 낭비
- [ ] **`name="None"` 엔티티와 `→ None` edge가 ~10% 토큰 낭비**
- 원인: finderlpg DB에 name 속성이 null인 Entity 노드 다수 존재
- 해결: `fetch_lpg_subgraph()` Cypher에 `WHERE n.name IS NOT NULL AND n.name <> 'None'` 추가
- 예상 효과: LPG effective token ratio 93% → ~100%
- 관련 파일: `src/utils/neo4j_client.py`

### Hub 지배 구조로 인한 Context 관련성 저하
- [ ] **"The Company" (degree=3,122, hub score=0.991)가 거의 모든 서브그래프에 포함**
- 원인: 극단적 star topology — LPG/RDF 모두 단일 허브가 그래프 지배
- 해결: Hub node pruning 또는 PCST 기반 subgraph selection (`example_codebase/compute_pcst.py` 참고)
- 예상 효과: 질문별 context 관련성 향상, Graph vs Text 격차 축소
- 관련 파일: `src/run_experiment.py`, `src/kvcache_experiment.py`

---

## High — 실험 인프라

### KV Cache Prefix Reuse 미작동
- [ ] **전 실험 구간에서 `gpu_prefix_cache_hit_rate = 0.0` — cold/warm speedup ≈ 1.0**
- 가능 원인: vLLM `--enable-prefix-caching` 플래그 미설정, 또는 reasoning model CoT 패턴이 prefix match 방해
- 조치: vLLM 서버 설정 확인, 필요 시 LMCache 레이어 활성화
- 관련 결과: `results/kvcache_experiment/20260208_091053/summary.json`

### EM(Exact Match) 전 조건 0.0
- [ ] **모든 context 조건에서 EM = 0.0**
- 원인: Reasoning model의 CoT 출력이 긴 서술형 → ground truth와 정확 매칭 불가
- 조치: 응답 후처리로 최종 답변만 추출하는 파싱 로직 추가, 또는 non-reasoning model 비교 실험
- 관련 파일: `src/utils/evaluation.py`

---

## Medium — 기능 개선

### Attention 실험 미실행
- [ ] **코드 구현 완료, GPU(A100) 환경에서 실행 필요**
- 구현 상태: `src/attention_experiment.py` + `AttentionExtractor` + `AttentionAnalyzer` 완성
- 5가지 조건: lpg_structured, lpg_natural, rdf_raw, rdf_cleaned, no_context
- 설계 문서: `docs/design/attention_experiment_design.md`

### Hard Prompt 실험 미구현
- [ ] **GNN/KGE 임베딩을 LLM에 직접 주입하는 방식 미검증**
- 현재: Soft prompt (텍스트 context)만 사용 — 학습된 임베딩은 서브그래프 추출에만 활용
- 참고: 7개 학습 완료 모델 (`results/checkpoints/`, 138MB)
- 관련 코드: `src/models.py`

### BERTScore 메트릭 미적용
- [ ] **Evaluator에 BERTScore 구현되어 있으나 KV Cache 실험에서 `--no-bertscore`로 비활성화**
- 이유: 실행 시간 증가 (per-question BERTScore 계산 비용)
- 조치: 차기 실험에서 활성화 또는 별도 후처리로 계산
- 관련 파일: `src/utils/evaluation.py`

### rdf_cleaned 조건 KV Cache 실험 미적용
- [ ] **`format_rdf_cleaned()`가 구현되어 있으나 KV Cache 실험의 `build_context()`에 미적용**
- 해결: `rdf_cleaned` 조건을 6번째 condition으로 추가하여 prefix 제거 효과 정량 측정
- 관련 파일: `src/kvcache_experiment.py`, `docs/design/attention_experiment_design.md`

---

## Low — 코드 품질 / 미래 과제

### rdf:type 트리플 중복
- [ ] **finderrdf에 rdf:type 트리플 341개가 별도 존재 — entity type 정보에 불필요한 토큰 소비**
- 조치: 서브그래프 추출 시 `rdf:type` 관계 필터링 옵션 추가

### Adaptive Condition Routing
- [ ] **질문 유형별 최적 context 자동 선택 미구현**
- 근거: Per-question 분석에서 45%는 graph > text, 51%는 text > graph
- 토픽별 차이: CapEx/Investment → LPG 우세, HR/Labor → Text 우세
- 조치: 질문 분류기 → condition routing 파이프라인 설계

### Non-Reasoning Model 비교
- [ ] **현재 reasoning model(CoT)만 사용 — vanilla latency가 가장 느린 역설**
- 원인: Context 없으면 더 긴 reasoning chain 생성 → 더 많은 completion tokens
- 조치: Non-CoT model로 동일 실험 실행하여 순수 context 효과 분리

---

## Resolved

*완료된 항목을 아래로 이동하고 완료 날짜와 관련 커밋/PR을 기록한다.*

- [x] **Attention 실험 코드 구현** (2026-02-10) — `src/attention_experiment.py`, `src/utils/attention.py`, `src/utils/attention_analysis.py`
- [x] **RDF prefix cleaner 구현** (2026-02-09) — `src/utils/formatting.py` `clean_rdf_uri()`, `format_rdf_cleaned()`
- [x] **Per-head attention 추출** (2026-02-09) — `AttentionConfig.aggregate_heads="none"` 옵션
- [x] **GNN/KGE 모델 7종 학습 완료** (2026-02-07) — GAT, GCN, GT, TransE, DistMult, ComplEx, RotatE
- [x] **공통 질문 ID 추출** (2026-02-07) — `data/processed/common_question_ids.json` (1,332개)
- [x] **KV Cache 실험 완료** (2026-02-08) — 5 conditions × 50 questions, 분석 노트북 작성

---

*마지막 갱신: 2026-02-12*
