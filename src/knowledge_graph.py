"""
KnowledgeGraph — Neo4j Cypher + 메모리 기반 BFS 폴백
논문 Section III.2 (2) 구현
최대 3-hop BFS 탐색
"""
from typing import List, Dict, Tuple, Optional
from collections import deque


class KnowledgeGraph:
    """
    노드/엣지 딕셔너리 기반 그래프 + BFS 3-hop 탐색
    Neo4j 연결 시 Cypher 쿼리로 자동 전환
    """

    def __init__(self, neo4j_uri: Optional[str] = None,
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "password"):
        self.nodes: Dict[str, Dict] = {}          # node_id → {name, type, props}
        self.edges: List[Tuple] = []              # (src, rel, dst)
        self.adj: Dict[str, List] = {}            # adjacency list
        self.neo4j_driver = None

        if neo4j_uri:
            self._connect_neo4j(neo4j_uri, neo4j_user, neo4j_password)

    def _connect_neo4j(self, uri, user, password):
        try:
            from neo4j import GraphDatabase
            self.neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
            print(f"✅ Neo4j 연결: {uri}")
        except Exception as e:
            print(f"⚠️  Neo4j 연결 실패 (메모리 모드): {e}")

    def add_node(self, node_id: str, name: str, node_type: str, **props):
        self.nodes[node_id] = {'name': name, 'type': node_type, **props}
        self.adj.setdefault(node_id, [])

    def add_edge(self, src: str, relation: str, dst: str):
        self.edges.append((src, relation, dst))
        self.adj.setdefault(src, []).append((relation, dst))
        self.adj.setdefault(dst, []).append((f"inv_{relation}", src))

    def search(self, query: str, top_k: int = 3, max_hops: int = 3) -> List[str]:
        """쿼리 키워드 매칭 → BFS 서브그래프 추출"""
        if self.neo4j_driver:
            return self._neo4j_search(query, top_k)
        return self._bfs_search(query, top_k, max_hops)

    def _bfs_search(self, query: str, top_k: int, max_hops: int) -> List[str]:
        # 시작 노드: 쿼리 키워드가 포함된 노드
        seeds = [nid for nid, info in self.nodes.items()
                 if info['name'] in query or query in info['name']]
        if not seeds:
            # 폴백: 모든 노드 중 관련 있는 것
            seeds = list(self.nodes.keys())[:3]

        visited, results = set(), []
        queue = deque([(s, 0) for s in seeds])

        while queue and len(results) < top_k * 10:
            node_id, depth = queue.popleft()
            if node_id in visited or depth > max_hops:
                continue
            visited.add(node_id)
            node = self.nodes.get(node_id, {})
            # 이 노드에서 나가는 엣지로 문장 생성
            for rel, neighbor in self.adj.get(node_id, []):
                if not rel.startswith('inv_'):
                    n_info = self.nodes.get(neighbor, {})
                    results.append(
                        f"{node.get('name','?')} --[{rel}]--> {n_info.get('name','?')}"
                    )
            if depth < max_hops:
                for _, neighbor in self.adj.get(node_id, []):
                    queue.append((neighbor, depth + 1))

        return results[:top_k]

    def _neo4j_search(self, query: str, top_k: int) -> List[str]:
        cypher = """
        MATCH (n)-[r]->(m)
        WHERE n.name CONTAINS $query OR m.name CONTAINS $query
        RETURN n.name + ' --[' + type(r) + ']--> ' + m.name AS path
        LIMIT $limit
        """
        with self.neo4j_driver.session() as session:
            result = session.run(cypher, query=query, limit=top_k)
            return [record['path'] for record in result]

    def load_university_data(self):
        """대학 행정 데이터 로드 — 60개 학과 / ~600명 교수 / ~1,500개 과목 / 400개 프로젝트"""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from data.dataset_generator import generate_university_data

        data = generate_university_data(seed=42)

        # ── 학과 노드 ──────────────────────────────────────────────
        for i, dept in enumerate(data["depts"]):
            self.add_node(f"d{i}", dept, "Department")

        dept_nid = {dept: f"d{i}" for i, dept in enumerate(data["depts"])}

        # ── 교수 노드 ──────────────────────────────────────────────
        for prof in data["professors"]:
            self.add_node(prof["id"], prof["name"], "Professor",
                          dept=prof["dept"], age=prof["age"],
                          research=prof["research"])

        prof_nid = {p["name"]: p["id"] for p in data["professors"]}

        # ── 과목 노드 ──────────────────────────────────────────────
        for i, course in enumerate(data["courses"]):
            self.add_node(f"c{i}", course["name"], "Course",
                          dept=course["dept"])

        course_nid = {c["name"]: f"c{i}" for i, c in enumerate(data["courses"])}

        # ── 프로젝트 노드 ──────────────────────────────────────────
        for i, (pname, _) in enumerate(data["projects"].items()):
            self.add_node(f"pr{i}", pname, "Project")

        proj_nid = {pname: f"pr{i}" for i, pname in enumerate(data["projects"].keys())}

        # ── 엣지: 소속 (교수 → 학과) ──────────────────────────────
        for prof in data["professors"]:
            pid = prof["id"]
            did = dept_nid.get(prof["dept"])
            if did:
                self.add_edge(pid, "소속", did)

        # ── 엣지: 담당 (교수 → 과목) ──────────────────────────────
        for prof in data["professors"]:
            for cname in prof["courses"]:
                cid = course_nid.get(cname)
                if cid:
                    self.add_edge(prof["id"], "담당", cid)

        # ── 엣지: 협력 (교수 → 교수) ──────────────────────────────
        added_collab = set()
        for prof in data["professors"]:
            for cname in prof["collab"]:
                pair = tuple(sorted([prof["name"], cname]))
                if pair not in added_collab:
                    cid2 = prof_nid.get(cname)
                    if cid2:
                        self.add_edge(prof["id"], "협력", cid2)
                    added_collab.add(pair)

        # ── 엣지: 참여 (교수 → 프로젝트) ─────────────────────────
        for pname, pprofs in data["projects"].items():
            prid = proj_nid.get(pname)
            if prid:
                for pname2 in pprofs:
                    pid2 = prof_nid.get(pname2)
                    if pid2:
                        self.add_edge(pid2, "참여", prid)

        # ── 엣지: 개설 (과목 → 학과) ──────────────────────────────
        for i, course in enumerate(data["courses"]):
            cid = f"c{i}"
            did = dept_nid.get(course["dept"])
            if did:
                self.add_edge(cid, "개설", did)

        print(f"KnowledgeGraph: 노드 {len(self.nodes)}개, 엣지 {len(self.edges)}개")
