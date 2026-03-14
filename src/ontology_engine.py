"""
OntologyEngine — OWL/Owlready2 기반 규범 레이어
논문 Section III.2 (3) 구현
클래스 계층, 제약 조건, 추론 규칙 포함
"""
from typing import List


class OntologyEngine:
    """
    전임/비전임 교수 계층 구조 + 연령/직급 논리 제약
    Owlready2 없을 시 규칙 기반 폴백으로 자동 전환
    """

    def __init__(self):
        self.onto = None
        self.use_owlready = False
        self._rules: List[dict] = []      # 폴백 규칙
        self._instances: List[dict] = []  # 폴백 인스턴스

        try:
            import owlready2
            self.use_owlready = True
            self._build_ontology()
        except ImportError:
            print("⚠️  owlready2 없음 → 규칙 기반 모드")
            self._load_rule_based()

    def _build_ontology(self):
        from owlready2 import get_ontology, Thing, ObjectProperty, DataProperty, sync_reasoner
        onto = get_ontology("http://university.org/triple_hybrid.owl#")
        with onto:
            class Person(Thing): pass
            class Professor(Person): pass
            class FullProfessor(Professor): pass
            class AdjunctProfessor(Professor): pass
            class Course(Thing): pass
            class Department(Thing): pass

            class teaches(ObjectProperty):
                domain = [Professor]; range = [Course]
            class belongsTo(ObjectProperty):
                domain = [Professor]; range = [Department]
            class collaboratesWith(ObjectProperty):
                domain = [Professor]; range = [Professor]
            class hasName(DataProperty):
                domain = [Thing]; range = [str]
            class hasAge(DataProperty):
                domain = [Person]; range = [int]

        self.onto = onto
        self._populate()
        print("✅ OntologyEngine (Owlready2) 초기화 완료")

    def _populate(self):
        """대학 행정 인스턴스 데이터 로드"""
        onto = self.onto
        with onto:
            from owlready2 import Thing
            data = [
                ("김철수", "FullProfessor", 45, "컴퓨터공학과", ["인공지능개론","딥러닝","강화학습"]),
                ("이영희", "FullProfessor", 38, "인공지능학과",  ["딥러닝","컴퓨터비전"]),
                ("박민수", "FullProfessor", 52, "컴퓨터공학과", ["자연어처리"]),
                ("정수진", "AdjunctProfessor", 36, "인공지능학과", ["컴퓨터비전"]),
            ]
            for name, ptype, age, dept, courses in data:
                cls = onto[ptype]
                if cls:
                    inst = cls(name.replace(" ","_"))
                    inst.hasName = [name]
                    inst.hasAge  = [age]

    def _load_rule_based(self):
        """Owlready2 없을 때 폴백 인스턴스+규칙"""
        self._instances = [
            {"name":"김철수","type":"FullProfessor","age":45,"dept":"컴퓨터공학과","courses":["인공지능개론","딥러닝","강화학습"]},
            {"name":"이영희","type":"FullProfessor","age":38,"dept":"인공지능학과","courses":["딥러닝","컴퓨터비전"]},
            {"name":"박민수","type":"FullProfessor","age":52,"dept":"컴퓨터공학과","courses":["자연어처리"]},
            {"name":"정수진","type":"AdjunctProfessor","age":36,"dept":"인공지능학과","courses":["컴퓨터비전"]},
        ]
        self._rules = [
            {"condition": lambda i, q: i["name"] in q,
             "result":    lambda i: f"{i['name']}은(는) {i['dept']} 소속, {i['age']}세, 담당과목: {', '.join(i['courses'])}"},
            {"condition": lambda i, q: any(c in q for c in i["courses"]),
             "result":    lambda i: f"{i['name']} 교수가 {[c for c in i['courses'] if c in i.get('_q','')]!r} 담당"},
            {"condition": lambda i, q: "이하" in q and any(str(n) in q for n in range(10,100))
                          and i["age"] <= int(next((x for x in q.split() if x.isdigit()), "99")),
             "result":    lambda i: f"{i['name']} ({i['age']}세) 해당"},
        ]

    def search(self, query: str, top_k: int = 3) -> List[str]:
        if self.use_owlready and self.onto:
            return self._owlready_search(query, top_k)
        return self._rule_search(query, top_k)

    def _rule_search(self, query: str, top_k: int) -> List[str]:
        results = []
        for inst in self._instances:
            inst["_q"] = query
            for rule in self._rules:
                try:
                    if rule["condition"](inst, query):
                        results.append(rule["result"](inst))
                        break
                except Exception:
                    pass
        if not results:
            results = [f"{i['name']}: {i['dept']}, {i['age']}세" for i in self._instances]
        return results[:top_k]

    def _owlready_search(self, query: str, top_k: int) -> List[str]:
        results = []
        for inst in self.onto.individuals():
            name_list = getattr(inst, 'hasName', [])
            name = name_list[0] if name_list else inst.name
            if name and (name in query or query in name):
                age_list = getattr(inst, 'hasAge', [])
                age = age_list[0] if age_list else "N/A"
                results.append(f"{name}: {type(inst).__name__}, {age}세")
        if not results:
            results = [
                f"{i.hasName[0] if i.hasName else i.name}"
                for i in list(self.onto.individuals())[:top_k]
            ]
        return results[:top_k]

    def check_constraint(self, entity_name: str, constraint: str) -> bool:
        """제약 조건 검증 (이하/이상/미만/초과)"""
        import re
        m = re.search(r'(\d+)\s*세\s*(이하|미만|이상|초과)', constraint)
        if not m:
            return True
        threshold, op = int(m.group(1)), m.group(2)
        inst = next((i for i in self._instances if i["name"] == entity_name), None)
        if not inst:
            return True
        age = inst["age"]
        if op == "이하":   return age <= threshold
        if op == "미만":   return age <  threshold
        if op == "이상":   return age >= threshold
        if op == "초과":   return age >  threshold
        return True
