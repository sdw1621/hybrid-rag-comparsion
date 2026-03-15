"""
QueryAnalyzer — 자연어 질의 분석기
논문 Section III.3 / Table 3 구현
"""
import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class QueryIntent:
    """질의 분석 결과 구조체"""
    query_type: str           # 'simple' | 'multi_hop' | 'conditional'
    entities: List[str]
    relations: List[str]
    constraints: List[str]
    complexity_score: float
    c_e: float = 0.0          # Entity Density    (수식 2)
    c_r: float = 0.0          # Relation Density  (수식 3)
    c_c: float = 0.0          # Constraint Density(수식 4)


class QueryAnalyzer:
    """
    NER + Relation Extraction + Constraint Analysis + Query Type Classification
    논문 Table 3 완전 구현
    """
    N_MAX_ENTITY     = 5
    N_MAX_RELATION   = 4
    N_MAX_CONSTRAINT = 3

    ENTITY_PATTERNS = [
        r'[가-힣]{2,4}\s*교수',
        r'[가-힣]{2,4}\s*학생',
        r'[가-힣]+(?:공학과|학과|학부|대학원)',
        r'[가-힣]+(?:프로젝트|연구|과목|수업)',
        r'(?:AI|ML|NLP|CV|DL|RL|RAG)',
    ]
    RELATION_KEYWORDS = [
        '소속', '협력', '담당', '참여', '지도',
        '가르치는', '수강', '공동', '연구하는', '담당하는',
    ]
    CONSTRAINT_PATTERNS = [
        r'\d+\s*세\s*(?:이하|미만|이상|초과)',
        r'(?:제외|excluding|except)',
        r'(?:이상|이하|초과|미만)',
        r'(?:and|or|AND|OR|그리고|또는)',
    ]

    def analyze(self, query: str) -> QueryIntent:
        entities    = self._extract_entities(query)
        relations   = self._extract_relations(query)
        constraints = self._extract_constraints(query)
        query_type  = self._classify(query, entities, relations, constraints)
        complexity  = min((len(entities)*0.3 + len(relations)*0.4 + len(constraints)*0.3) / 3.0, 1.0)
        c_e = min(len(entities)    / self.N_MAX_ENTITY,     1.0)
        c_r = min(len(relations)   / self.N_MAX_RELATION,   1.0)
        c_c = min(len(constraints) / self.N_MAX_CONSTRAINT, 1.0)
        return QueryIntent(
            query_type=query_type, entities=entities,
            relations=relations, constraints=constraints,
            complexity_score=complexity, c_e=c_e, c_r=c_r, c_c=c_c,
        )

    def _extract_entities(self, query):
        found = []
        for p in self.ENTITY_PATTERNS:
            found.extend(re.findall(p, query))
        return list(set(found))

    def _extract_relations(self, query):
        return [kw for kw in self.RELATION_KEYWORDS if kw in query]

    def _extract_constraints(self, query):
        found = []
        for p in self.CONSTRAINT_PATTERNS:
            found.extend(re.findall(p, query))
        return list(set(found))

    def _classify(self, query, entities, relations, constraints):
        if constraints:
            return 'conditional'
        if len(entities) >= 2 or len(relations) >= 2:
            return 'multi_hop'
        return 'simple'
