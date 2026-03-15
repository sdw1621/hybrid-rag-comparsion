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
        """논문 실험용 대학 행정 데이터 로드"""
        # 노드
        professors = [
            ("p1","김철수","Professor",{"dept":"컴퓨터공학과","age":45}),
            ("p2","이영희","Professor",{"dept":"인공지능학과","age":38}),
            ("p3","박민수","Professor",{"dept":"컴퓨터공학과","age":52}),
            ("p4","정수진","Professor",{"dept":"인공지능학과","age":36}),
        ]
        for nid, name, ntype, props in professors:
            self.add_node(nid, name, ntype, **props)

        courses = [
            ("c1","인공지능개론","Course"),
            ("c2","딥러닝","Course"),
            ("c3","자연어처리","Course"),
            ("c4","컴퓨터비전","Course"),
            ("c5","강화학습","Course"),
        ]
        for nid, name, ntype in courses:
            self.add_node(nid, name, ntype)

        depts = [
            ("d1","컴퓨터공학과","Department"),
            ("d2","인공지능학과","Department"),
        ]
        for nid, name, ntype in depts:
            self.add_node(nid, name, ntype)

        projs = [
            ("pr1","AI융합프로젝트","Project"),
            ("pr2","NLP연구프로젝트","Project"),
        ]
        for nid, name, ntype in projs:
            self.add_node(nid, name, ntype)

        # 엣지
        edges = [
            ("p1","담당","c1"), ("p1","담당","c2"), ("p1","담당","c5"),
            ("p2","담당","c2"), ("p2","담당","c4"),
            ("p3","담당","c3"),
            ("p4","담당","c4"),
            ("p1","소속","d1"), ("p3","소속","d1"),
            ("p2","소속","d2"), ("p4","소속","d2"),
            ("p1","협력","p2"), ("p2","협력","p3"),
            ("p1","참여","pr1"), ("p2","참여","pr1"),
            ("p3","참여","pr2"),
        ]
        for s, r, d in edges:
            self.add_edge(s, r, d)

        print(f"✅ KnowledgeGraph: 노드 {len(self.nodes)}개, 엣지 {len(self.edges)}개")
