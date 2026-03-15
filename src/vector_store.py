"""
VectorStore — FAISS 기반 벡터 검색
논문 Section III.2 (1) / Table 7, 8 구현
"""
from typing import List, Tuple
import numpy as np

try:
    import faiss
    from langchain_openai import OpenAIEmbeddings
    from langchain_core.documents import Document
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class VectorStore:
    """
    OpenAI text-embedding-3-small + FAISS IndexFlatIP (cosine)
    chunk_size=1000, overlap=200
    """

    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        self.embedding_model = embedding_model
        self.embeddings = None
        self.index = None
        self.documents: List[str] = []
        self._dim = 1536

    def build(self, documents: List[str]):
        """문서 임베딩 후 FAISS 인덱스 구축"""
        from langchain_openai import OpenAIEmbeddings
        import faiss

        self.documents = documents
        self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
        vecs = self.embeddings.embed_documents(documents)
        mat = np.array(vecs, dtype='float32')
        # L2 정규화 → 내적 = 코사인 유사도
        faiss.normalize_L2(mat)
        self.index = faiss.IndexFlatIP(mat.shape[1])
        self.index.add(mat)
        print(f"✅ VectorStore 구축: {len(documents)}개 문서, dim={mat.shape[1]}")

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """코사인 유사도 기반 top-k 검색"""
        import faiss
        q_vec = np.array(
            self.embeddings.embed_query(query), dtype='float32'
        ).reshape(1, -1)
        faiss.normalize_L2(q_vec)
        scores, idxs = self.index.search(q_vec, top_k)
        return [(self.documents[i], float(scores[0][j]))
                for j, i in enumerate(idxs[0]) if i >= 0]
