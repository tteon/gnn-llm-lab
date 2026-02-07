# Neo4j 스키마 및 쿼리 패턴

## finderlpg (LPG Database)

### 노드

```
(:Entity {
    id: String,
    label: String,
    name: String,
    question_ids: [String],   -- 이 엔티티와 관련된 질문 ID 리스트
    ...properties
})
```

### 엣지

```
(:Entity)-[r:RELATION_TYPE]->(:Entity)
```

- 관계 타입은 다양 (금융 도메인 관계)
- `type(r)` 로 관계명 접근

### 서브그래프 쿼리 패턴

```cypher
-- question_id로 시드 엔티티 찾기
MATCH (e:Entity)
WHERE $qid IN e.question_ids
WITH collect(e) as seeds

-- max_hops만큼 확장
UNWIND seeds as seed
OPTIONAL MATCH path = (seed)-[*1..2]-(connected:Entity)
WITH seeds + collect(DISTINCT connected) as all_raw
WITH [n IN all_raw WHERE n IS NOT NULL] as all_nodes

-- 중복 제거 후 엣지 수집
UNWIND all_nodes as n
WITH collect(DISTINCT n) as all_nodes
UNWIND all_nodes as n
OPTIONAL MATCH (n)-[r]->(m:Entity)
WHERE m IN all_nodes
RETURN collect(DISTINCT { id: n.id, label: n.label, name: coalesce(n.name, n.id), properties: properties(n) }) as nodes,
       collect(DISTINCT { source: startNode(r).id, target: endNode(r).id, type: type(r) }) as edges
```

**주의**: `WITH DISTINCT n` 사용 시 이전 `WITH`에서 정의한 `all_nodes`가 scope에서 사라짐.
`UNWIND → collect(DISTINCT) → UNWIND` 패턴으로 해결.

---

## finderrdf (RDF Database)

### 노드

```
(:Resource {
    uri: String,    -- FIBO URI (e.g., "https://spec.edmcouncil.org/fibo/...")
    ...literal properties
})
```

### 엣지

```
(:Resource)-[:TRIPLE {
    predicate: String,       -- RDF predicate URI
    question_id: String      -- 이 triple과 관련된 질문 ID
}]->(:Resource)
```

- 모든 관계가 `:TRIPLE` 타입 (RDF reification)
- 실제 관계 의미는 `rel.predicate` 속성에 저장
- 질문 연결은 `rel.question_id` 속성으로

### 서브그래프 쿼리 패턴

```cypher
-- question_id로 관련 triple 찾기
MATCH (r:Resource)-[rel:TRIPLE]->(r2:Resource)
WHERE rel.question_id = $qid
RETURN collect(DISTINCT {id: r.uri, uri: r.uri})
     + collect(DISTINCT {id: r2.uri, uri: r2.uri}) as nodes,
       collect(DISTINCT {source: r.uri, target: r2.uri, type: rel.predicate}) as edges
```

**주의**: `type(rel)`은 항상 `"TRIPLE"`을 반환하므로, 실제 predicate는 `rel.predicate`로 접근해야 함.

---

## LPG vs RDF 주요 차이

| 항목 | finderlpg | finderrdf |
|------|-----------|-----------|
| 질문 연결 | 노드의 `question_ids` 배열 | 엣지의 `question_id` 속성 |
| 관계 타입 | Neo4j relationship type (`type(r)`) | `rel.predicate` 속성 (URI) |
| 노드 식별자 | `n.id` | `r.uri` |
| 확장 전략 | Seed → hop expansion | Direct triple match |
| 토큰 효율 | 높음 (간결한 이름) | 낮음 (FIBO URI prefix 포함) |
