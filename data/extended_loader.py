"""
확장 데이터 로더
knowledge_graph.py 의 load_university_data() 대체
문서 생성기 (200건) 포함
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.university_data import (
    DEPARTMENTS, PROFESSORS, COURSES, PROJECTS, COLLABORATIONS,
    get_prof_courses, get_dept_profs
)


def load_extended_graph(kg):
    """KnowledgeGraph 인스턴스에 확장 데이터 로드"""
    # ── 노드 추가 ──────────────────────────────────────────
    for d in DEPARTMENTS:
        kg.add_node(d["id"], d["name"], "Department",
                    code=d["code"], college=d["college"])

    for p in PROFESSORS:
        kg.add_node(p["id"], p["name"], "Professor",
                    age=p["age"], rank=p["rank"], dept=p["dept"],
                    email=p["email"])

    for c in COURSES:
        kg.add_node(c["id"], c["name"], "Course",
                    credits=c["credits"], dept=c["dept"])

    for pr in PROJECTS:
        kg.add_node(pr["id"], pr["name"], "Project", field=pr["field"])

    # ── 엣지 추가 ──────────────────────────────────────────
    # 교수 → 학과 소속
    dept_map = {d["name"]: d["id"] for d in DEPARTMENTS}
    for p in PROFESSORS:
        did = dept_map.get(p["dept"])
        if did:
            kg.add_edge(p["id"], "소속", did)

    # 교수 → 과목 담당
    for c in COURSES:
        for pid in c["profs"]:
            kg.add_edge(pid, "담당", c["id"])

    # 교수 ↔ 프로젝트 참여
    for pr in PROJECTS:
        for pid in pr["profs"]:
            kg.add_edge(pid, "참여", pr["id"])

    # 교수 ↔ 교수 협력
    for p1, p2 in COLLABORATIONS:
        kg.add_edge(p1, "협력", p2)

    # 과목 → 학과 개설
    for c in COURSES:
        did = dept_map.get(c["dept"])
        if did:
            kg.add_edge(c["id"], "개설", did)

    node_cnt = len(kg.nodes)
    edge_cnt = len(kg.edges)
    print(f"✅ KnowledgeGraph 확장 로드: 노드 {node_cnt}개, 엣지 {edge_cnt}개")
    return kg


def generate_documents() -> list:
    """
    200건 자연어 문서 생성
    교수 프로필 / 과목 설명 / 학과 소개 / 프로젝트 소개
    """
    docs = []

    # 교수별 프로필 문서 (30건 × 2 = 60건)
    for p in PROFESSORS:
        courses = get_prof_courses(p["id"])
        course_names = ", ".join(c["name"] for c in courses) if courses else "없음"
        research = ", ".join(p["research"])

        docs.append(
            f"{p['name']} 교수는 {p['age']}세이며 {p['dept']} {p['rank']}이다. "
            f"이메일은 {p['email']}이다. "
            f"주요 연구 분야는 {research}이며, "
            f"담당 과목은 {course_names}이다."
        )
        docs.append(
            f"{p['name']} 교수({p['dept']})는 {research} 분야를 전문으로 연구한다. "
            f"{p['rank']}으로 재직 중이며 나이는 {p['age']}세이다. "
            f"{'젊은 연구자로 활발한 연구 활동을 이어가고 있다.' if p['age'] < 40 else '오랜 경험을 바탕으로 깊이 있는 연구를 수행한다.'}"
        )

    # 과목별 설명 문서 (40건 × 2 = 80건)
    course_descs = {
        "인공지능개론":   "AI의 기초 개념, 탐색, 지식표현, 기계학습 입문을 다룬다.",
        "딥러닝":         "신경망 구조, CNN, RNN, Transformer, 역전파 알고리즘을 다룬다.",
        "자연어처리":     "텍스트 전처리, 언어 모델, 감성 분석, 기계 번역을 다룬다.",
        "알고리즘":       "정렬, 탐색, 동적 프로그래밍, 그래프 알고리즘을 다룬다.",
        "강화학습":       "MDP, Q-Learning, Policy Gradient, PPO, 게임 AI를 다룬다.",
        "분산시스템":     "분산 컴퓨팅 원리, 일관성, 가용성, 파티션 허용을 다룬다.",
        "클라우드컴퓨팅": "AWS, Azure, GCP, 컨테이너, 마이크로서비스를 다룬다.",
        "데이터베이스":   "관계형 DB, SQL, 트랜잭션, 인덱싱, NoSQL을 다룬다.",
        "운영체제":       "프로세스, 메모리 관리, 파일 시스템, 동기화를 다룬다.",
        "컴퓨터네트워크": "TCP/IP, 라우팅, HTTP, 소켓 프로그래밍을 다룬다.",
        "컴퓨터비전":     "이미지 처리, 객체 탐지, 세그멘테이션, GAN을 다룬다.",
        "생성모델":       "VAE, GAN, Diffusion 모델의 이론과 구현을 다룬다.",
        "설명가능AI":     "LIME, SHAP, 어텐션 시각화, 모델 해석 방법론을 다룬다.",
        "지식그래프":     "RDF, OWL, SPARQL, Neo4j, 지식 표현 방법을 다룬다.",
        "로봇공학":       "로봇 기구학, 경로 계획, SLAM, ROS를 다룬다.",
        "자연어이해":     "BERT, GPT, 언어 이해 태스크, 파인튜닝을 다룬다.",
        "멀티모달AI":     "텍스트-이미지 모델, CLIP, DALL-E 원리를 다룬다.",
        "AI윤리":         "AI 공정성, 책임, 투명성, 사회적 영향을 다룬다.",
        "소프트웨어공학": "요구 분석, 설계 패턴, 테스트, 형상 관리를 다룬다.",
        "프로그래밍언어론":"문법, 의미론, 컴파일러 원리, 인터프리터를 다룬다.",
        "모바일프로그래밍":"Android, iOS, React Native, 모바일 UI/UX를 다룬다.",
        "DevOps실습":     "CI/CD, Docker, Kubernetes, 모니터링 도구를 다룬다.",
        "데이터마이닝":   "클러스터링, 분류, 연관 규칙, 이상 탐지를 다룬다.",
        "시계열분석":     "ARIMA, LSTM, Prophet, 예측 모델을 다룬다.",
        "빅데이터처리":   "Hadoop, Spark, Kafka, 분산 처리 아키텍처를 다룬다.",
        "데이터시각화":   "Matplotlib, D3.js, Tableau, 인포그래픽 원리를 다룬다.",
        "통계적학습":     "회귀, 분류, 정규화, 교차검증, 모델 선택을 다룬다.",
        "암호학":         "대칭/비대칭 암호, 해시, PKI, 양자 암호를 다룬다.",
        "네트워크보안":   "방화벽, IDS, VPN, 취약점 스캐닝을 다룬다.",
        "블록체인":       "분산 원장, 스마트 컨트랙트, DeFi, NFT를 다룬다.",
        "신호처리":       "FFT, 필터 설계, 디지털 신호 처리를 다룬다.",
        "VLSI설계":       "CMOS 회로, 레이아웃 설계, EDA 도구를 다룬다.",
        "임베디드시스템": "ARM, RTOS, 펌웨어 개발, 하드웨어 인터페이스를 다룬다.",
        "최적화이론":     "볼록 최적화, 경사 하강법, 라그랑지안을 다룬다.",
        "확률론":         "확률 공간, 분포, 중심 극한 정리, 마르코프 체인을 다룬다.",
        "베이지안통계":   "사전/사후 분포, MCMC, 베이지안 네트워크를 다룬다.",
        "생존분석":       "생존 함수, Kaplan-Meier, Cox 비례 위험 모형을 다룬다.",
        "머신러닝기초":   "지도학습, 비지도학습, 특성 공학, 모델 평가를 다룬다.",
        "수치해석":       "오차 분석, 수치 적분, 미분 방정식 풀이를 다룬다.",
        "선형대수학":     "행렬 연산, 고유값, SVD, 선형 변환을 다룬다.",
    }

    for c in COURSES:
        desc = course_descs.get(c["name"], f"{c['name']} 관련 심화 내용을 다룬다.")
        profs = ", ".join(
            p["name"] for p in PROFESSORS if p["id"] in c["profs"]
        )
        docs.append(
            f"{c['name']} 과목은 {c['dept']}에서 개설된 {c['credits']}학점 과목이다. "
            f"{profs} 교수가 담당한다. {desc}"
        )
        docs.append(
            f"{c['name']}은 {c['dept']} 소속 과목으로 {c['credits']}학점이다. "
            f"담당 교수: {profs}."
        )

    # 학과 소개 문서 (8건 × 2 = 16건)
    for d in DEPARTMENTS:
        dept_profs = get_dept_profs(d["name"])
        prof_names = ", ".join(p["name"] for p in dept_profs)
        researches = list({r for p in dept_profs for r in p["research"]})[:4]
        docs.append(
            f"{d['name']}은 {d['college']} 소속 학과이다. "
            f"소속 교수는 {prof_names}이다. "
            f"주요 연구 분야: {', '.join(researches)}."
        )
        docs.append(
            f"{d['college']} {d['name']}(학과코드: {d['code']})에는 "
            f"총 {len(dept_profs)}명의 교수가 재직 중이다. "
            f"정교수, 부교수, 조교수로 구성되어 있으며 다양한 연구를 수행한다."
        )

    # 프로젝트 소개 문서 (15건 × 2 = 30건)
    for pr in PROJECTS:
        profs = ", ".join(
            p["name"] for p in PROFESSORS if p["id"] in pr["profs"]
        )
        docs.append(
            f"{pr['name']}은 {pr['field']} 분야 연구 프로젝트이다. "
            f"참여 교수: {profs}."
        )
        docs.append(
            f"{pr['field']} 분야의 {pr['name']} 프로젝트에는 "
            f"{profs} 교수가 공동 참여하고 있다."
        )

    print(f"✅ 문서 생성: {len(docs)}건")
    return docs


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from src.knowledge_graph import KnowledgeGraph
    kg = KnowledgeGraph()
    load_extended_graph(kg)
    docs = generate_documents()
    print(f"총 문서: {len(docs)}건")
