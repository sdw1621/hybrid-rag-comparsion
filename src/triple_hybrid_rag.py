"""
TripleHybridRAG — Vector + Graph + Ontology 통합 파이프라인
논문 전체 아키텍처 구현 (Fig. 1)

S_total = α_final·S_vector + β_final·S_graph + γ_final·S_ontology
"""
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .query_analyzer import QueryAnalyzer, QueryIntent
from .dwa import DWA, DWAWeights
from .vector_store import VectorStore
from .knowledge_graph import KnowledgeGraph
from .ontology_engine import OntologyEngine


@dataclass
class RAGResult:
    """질의 처리 결과"""
    answer: str
    elapsed: float
    weights: DWAWeights
    intent: QueryIntent
    vector_contexts: List[str]
    graph_contexts: List[str]
    onto_contexts: List[str]
    prompt_used: str = ""


PROMPT_TEMPLATE = (
    "Based on the following context, answer the question accurately. "
    "If the answer cannot be determined from the context, state that the information is not available.\n\n"
    "Context:\n{context}\n\n"
    "Question: {query}\n\n"
    "Answer:"
)


class TripleHybridRAG:
    """
    메인 클래스
    - add_knowledge() : 지식 추가
    - query()         : 질의 처리 (DWA 가중치 자동 적용)
    """

    def __init__(self,
                 openai_api_key: Optional[str] = None,
                 neo4j_uri: Optional[str] = None,
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "password",
                 llm_model: str = "gpt-4o-mini",
                 temperature: float = 0.0,
                 top_k: int = 3,
                 lambda_: float = 0.3):

        import os
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key

        self.top_k       = top_k
        self.llm_model   = llm_model
        self.temperature = temperature

        # 컴포넌트 초기화
        self.analyzer = QueryAnalyzer()
        self.dwa       = DWA(lambda_=lambda_)
        self.vector    = VectorStore()
        self.graph     = KnowledgeGraph(neo4j_uri, neo4j_user, neo4j_password)
        self.ontology  = OntologyEngine()

        # LLM
        from langchain_openai import ChatOpenAI
        self.llm = ChatOpenAI(model=llm_model, temperature=temperature)

        self._documents: List[str] = []
        self._built = False

    # ── 지식 추가 ──────────────────────────────────────────
    def add_documents(self, docs: List[str]):
        self._documents.extend(docs)

    def add_graph_edge(self, src: str, relation: str, dst: str,
                       src_type: str = "Node", dst_type: str = "Node"):
        if src not in self.graph.nodes:
            self.graph.add_node(src, src, src_type)
        if dst not in self.graph.nodes:
            self.graph.add_node(dst, dst, dst_type)
        self.graph.add_edge(src, relation, dst)

    def load_university_sample(self):
        """논문 실험 데이터 일괄 로드 — 60개 학과 / ~600명 교수 / ~1,500개 과목 / 400개 프로젝트"""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from data.dataset_generator import generate_university_data

        self.graph.load_university_data()

        data    = generate_university_data(seed=42)
        docs    = []

        # ── 교수 소개 문서 (교수 1명당 1건) ─────────────────────────
        for prof in data["professors"]:
            courses_str = ", ".join(prof["courses"]) if prof["courses"] else "없음"
            collab_str  = ", ".join(prof["collab"])  if prof["collab"]  else "없음"
            docs.append(
                f"{prof['name']} 교수는 {prof['age']}세이며 {prof['dept']} 소속이다. "
                f"담당 과목: {courses_str}. "
                f"연구 분야: {prof['research']}. "
                f"협력 교수: {collab_str}."
            )

        # ── 학과 안내 문서 (학과 1개당 1건) ─────────────────────────
        for dept in data["depts"]:
            prof_names   = data["dept_profs"].get(dept, [])
            course_names = data["dept_courses"].get(dept, [])
            docs.append(
                f"{dept} 소속 교수는 총 {len(prof_names)}명이며, "
                f"개설 과목 수는 {len(course_names)}개다. "
                f"교수 목록: {', '.join(prof_names[:5])}{'...' if len(prof_names) > 5 else ''}. "
                f"대표 과목: {', '.join(course_names[:5])}{'...' if len(course_names) > 5 else ''}."
            )

        # ── 프로젝트 문서 (프로젝트 1개당 1건) ──────────────────────
        for pname, pprofs in data["projects"].items():
            docs.append(
                f"{pname} 프로젝트에는 {', '.join(pprofs)}가 참여한다. "
                f"총 {len(pprofs)}명의 교수가 참여하는 연구 과제다."
            )

        self.add_documents(docs)
        print(f"Vector 문서: {len(docs)}건 로드 완료 "
              f"(교수 {len(data['professors'])}건 + "
              f"학과 {len(data['depts'])}건 + "
              f"프로젝트 {len(data['projects'])}건)")

    def build(self):
        """벡터 인덱스 구축"""
        if self._documents:
            self.vector.build(self._documents)
        self._built = True
        print("✅ TripleHybridRAG 빌드 완료")

    # ── 질의 처리 ──────────────────────────────────────────
    def query(self, question: str) -> RAGResult:
        start = time.time()

        # 1. 질의 분석
        intent  = self.analyzer.analyze(question)
        weights = self.dwa.compute(intent)

        # 2. 병렬 검색 (각 소스)
        v_ctxs = [doc for doc, _ in self.vector.search(question, self.top_k)] if self._built else []
        g_ctxs = self.graph.search(question, self.top_k)
        o_ctxs = self.ontology.search(question, self.top_k)

        # 3. 가중치 기반 컨텍스트 통합
        context = self._merge_contexts(
            v_ctxs, g_ctxs, o_ctxs,
            weights.alpha, weights.beta, weights.gamma
        )

        # 4. LLM 답변 생성
        prompt = PROMPT_TEMPLATE.format(context=context, query=question)
        response = self.llm.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)

        elapsed = time.time() - start
        return RAGResult(
            answer=answer, elapsed=elapsed,
            weights=weights, intent=intent,
            vector_contexts=v_ctxs,
            graph_contexts=g_ctxs,
            onto_contexts=o_ctxs,
            prompt_used=prompt,
        )

    def _merge_contexts(self, v_ctxs, g_ctxs, o_ctxs,
                        alpha, beta, gamma) -> str:
        """가중치 비례 컨텍스트 조합"""
        total = alpha + beta + gamma
        n_v = max(1, round(self.top_k * alpha / total))
        n_g = max(1, round(self.top_k * beta  / total))
        n_o = max(1, round(self.top_k * gamma / total))

        parts = []
        if v_ctxs:
            parts.append(f"[Vector(α={alpha:.2f})]\n" + "\n".join(v_ctxs[:n_v]))
        if g_ctxs:
            parts.append(f"[Graph(β={beta:.2f})]\n"  + "\n".join(g_ctxs[:n_g]))
        if o_ctxs:
            parts.append(f"[Ontology(γ={gamma:.2f})]\n" + "\n".join(o_ctxs[:n_o]))
        return "\n\n".join(parts)
