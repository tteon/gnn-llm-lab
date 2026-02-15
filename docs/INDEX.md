# Documentation Index

> 프로젝트 문서를 MECE(Mutually Exclusive, Collectively Exhaustive) 카테고리로 분류한 마스터 인덱스.

---

## 읽기 가이드

| 목적 | 경로 |
|------|------|
| 프로젝트 개요 | [`README.md`](../README.md) → [`CLAUDE.md`](../CLAUDE.md) |
| 실험 설계 이해 | [`docs/design/experiment_design.md`](design/experiment_design.md) |
| 전체 결과 파악 | [`docs/analysis/comprehensive_experiment_report.md`](analysis/comprehensive_experiment_report.md) |
| 알려진 문제/개선 과제 | [`KNOWN_ISSUES.md`](../KNOWN_ISSUES.md) |

---

## 카테고리별 문서 맵

### Design — 실험 설계

실험의 가설, 조건, 메트릭을 정의하는 설계서. 실험 실행 전에 작성하고, 실행 후에는 수정하지 않는다.

| 문서 | 줄수 | 상태 | 요약 |
|------|------|------|------|
| [experiment_design.md](design/experiment_design.md) | 709 | 확정 | 메인 실험 설계 — RQ1~3, 6가지 조건(A~D + soft/hard), 품질/latency 메트릭 |
| [attention_experiment_design.md](design/attention_experiment_design.md) | 544 | 확정 | LPG vs RDF attention 실험 — 5 조건, 5 가설(H1~H5), attention/quality 메트릭, 구현 계획 |

### Analysis — 실험 결과 및 데이터 분석

실험 결과를 분석하고 인사이트를 도출한 문서. 실험 완료 후 작성.

| 문서 | 줄수 | 상태 | 요약 |
|------|------|------|------|
| [comprehensive_experiment_report.md](analysis/comprehensive_experiment_report.md) | 1,234 | 최신 | **종합 보고서** — 전체 실험 통합 분석 (KG, GNN/KGE, KV Cache, Attention, Legacy) |
| [kvcache_experiment_analysis.md](analysis/kvcache_experiment_analysis.md) | 1,098 | 완료 | KV Cache 실험 상세 분석 — 품질/latency/토큰/per-question 분석, 9개 차트 |
| [finder_kg_analysis.md](analysis/finder_kg_analysis.md) | 664 | 완료 | FinDER KG 데이터셋 특성 분석 — 노드/엣지 통계, 데이터 품질, 임베딩 전략 |
| [gnn_kge_focused_analysis.md](analysis/gnn_kge_focused_analysis.md) | 553 | 완료 | GNN vs KGE 모델-데이터 적합성 — LPG→GAT, RDF→TransE 근거 분석 |

### Reference — 기술 참조 자료

프로젝트에서 참조하는 기술 문서. 필요 시 열람.

| 문서 | 줄수 | 상태 | 요약 |
|------|------|------|------|
| [neo4j_schema.md](reference/neo4j_schema.md) | 102 | 안정 | Neo4j DB 스키마 — finderlpg/finderrdf 노드·엣지 구조, Cypher 쿼리 패턴 |
| [cugraph_pyglib_analysis.md](reference/cugraph_pyglib_analysis.md) | 506 | 안정 | GPU 가속 기술 선택 — torch.compile > pyg-lib > cuGraph (14K 노드에 cuGraph 불필요) |
| [cugraph_pyg_study_guide.md](reference/cugraph_pyg_study_guide.md) | 968 | 참조 | cuGraph-PyG 아키텍처 학습 가이드 — Remote Backend, GPU sampling, multi-GPU (미사용) |

### External — 외부 공유용

프로젝트 외부 공유를 위해 작성된 문서.

| 문서 | 줄수 | 상태 | 요약 |
|------|------|------|------|
| [linkedin_post_kvcache.md](external/linkedin_post_kvcache.md) | 80 | 완료 | KV Cache 실험 LinkedIn 포스트 — 핵심 발견 4가지 요약 |

---

## 프로젝트 루트 문서

| 문서 | 용도 | 대상 |
|------|------|------|
| [`README.md`](../README.md) | 프로젝트 소개, 아키텍처, Quick Start | 외부 / 신규 참여자 |
| [`CLAUDE.md`](../CLAUDE.md) | 개발 가이드라인, 파일 맵, 명령어 | Claude Code / 개발자 |
| [`KNOWN_ISSUES.md`](../KNOWN_ISSUES.md) | 알려진 문제 및 개선 백로그 (우선순위별) | 개발자 |

---

## 문서 상태 정의

| 상태 | 의미 |
|------|------|
| 확정 | 실험 설계가 확정되어 변경하지 않음 |
| 최신 | 최근 갱신됨, 활발히 유지보수 중 |
| 완료 | 분석이 완료되어 추가 수정 불필요 (결과 확정) |
| 안정 | 참조 자료로서 큰 변경 없이 유지 |
| 참조 | 직접 사용하지 않는 학습/참고 자료 |

---

## 문서 갱신 이력

| 날짜 | 변경 | 문서 |
|------|------|------|
| 2026-02-12 | 종합 보고서 작성 | `analysis/comprehensive_experiment_report.md` |
| 2026-02-12 | MECE 카테고리 정리, INDEX.md 생성 | `docs/INDEX.md` |
| 2026-02-12 | KNOWN_ISSUES.md 신규 생성 | `KNOWN_ISSUES.md` |
| 2026-02-08 | KV Cache 실험 분석 완료 | `analysis/kvcache_experiment_analysis.md` |
| 2026-02-08 | cuGraph 기술 분석 | `reference/cugraph_pyglib_analysis.md` |
| 2026-02-07 | 메인 실험 설계서 확정 | `design/experiment_design.md` |
| 2026-02-07 | KG 데이터 분석 | `analysis/finder_kg_analysis.md` |
| 2026-02-07 | GNN/KGE 적합성 분석 | `analysis/gnn_kge_focused_analysis.md` |

---

## 갱신 규칙

1. **새 실험 설계** → `docs/design/`에 `{실험명}_design.md` 생성
2. **실험 완료 분석** → `docs/analysis/`에 `{실험명}_analysis.md` 생성
3. **기술 참조 추가** → `docs/reference/`에 추가
4. **외부 공유 문서** → `docs/external/`에 추가
5. **문제 발견** → `KNOWN_ISSUES.md`에 적절한 priority 섹션에 추가
6. **문제 해결** → `KNOWN_ISSUES.md`의 해당 항목을 `Resolved` 섹션으로 이동, 날짜 기록
7. **모든 변경** → 이 INDEX.md의 갱신 이력 테이블에 기록
