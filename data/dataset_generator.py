"""
Gold QA Dataset Generator
논문 Section V.1 / Table 9, 10 구현
500개 질문-정답-참조근거 쌍 생성
Simple 40% / Multi-hop 35% / Conditional 25%
"""
import json
import random
from typing import List, Dict


def build_gold_dataset(seed: int = 42) -> List[Dict]:
    """
    500개 Gold QA 쌍 생성
    Returns: [{"id", "query", "answer", "reference", "type"}, ...]
    """
    random.seed(seed)
    dataset = []

    # ── Simple (200개) ────────────────────────────────────────
    simple_templates = [
        ("{prof} 교수의 소속 학과는?",          "{dept}",            "prof_dept"),
        ("{prof} 교수의 나이는?",               "{age}세",           "prof_age"),
        ("{prof} 교수가 담당하는 과목은?",       "{courses}",         "prof_courses"),
        ("{course} 과목을 담당하는 교수는?",     "{prof}",            "course_prof"),
        ("{dept}에 소속된 교수 목록은?",         "{dept_profs}",      "dept_profs"),
        ("{prof} 교수의 연구 분야는?",           "{research}",        "prof_research"),
        ("{course}은 몇 학점인가?",              "3학점",             "course_credit"),
        ("{proj} 프로젝트 참여 교수는?",         "{proj_profs}",      "proj_profs"),
    ]

    professors = [
        {"name":"김철수","dept":"컴퓨터공학과","age":45,
         "courses":["인공지능개론","딥러닝","강화학습"],"research":"머신러닝, 자연어처리"},
        {"name":"이영희","dept":"인공지능학과","age":38,
         "courses":["딥러닝","컴퓨터비전"],"research":"딥러닝, 컴퓨터비전"},
        {"name":"박민수","dept":"컴퓨터공학과","age":52,
         "courses":["자연어처리"],"research":"자연어처리, 정보검색"},
        {"name":"정수진","dept":"인공지능학과","age":36,
         "courses":["컴퓨터비전"],"research":"딥러닝, 컴퓨터비전"},
    ]
    courses   = ["인공지능개론","딥러닝","자연어처리","컴퓨터비전","강화학습"]
    depts     = {"컴퓨터공학과":["김철수","박민수"], "인공지능학과":["이영희","정수진"]}
    projects  = {"AI융합프로젝트":["김철수","이영희"], "NLP연구프로젝트":["박민수"]}

    idx = 1
    while len([d for d in dataset if d['type']=='simple']) < 200:
        prof   = random.choice(professors)
        course = random.choice(courses)
        dept   = random.choice(list(depts.keys()))
        proj   = random.choice(list(projects.keys()))
        tmpl, ans_tmpl, qtype = random.choice(simple_templates)

        q = tmpl.format(
            prof=prof['name'], dept=dept, course=course, proj=proj,
            courses=", ".join(prof['courses'])
        )
        a = ans_tmpl.format(
            dept=prof['dept'], age=prof['age'],
            courses=", ".join(prof['courses']),
            prof=prof['name'],
            dept_profs=", ".join(depts[dept]),
            research=prof['research'],
            proj_profs=", ".join(projects[proj]),
        )
        ref = f"{prof['name']} 교수 정보 / {dept} 학과 정보"
        dataset.append({"id": idx, "query": q, "answer": a,
                        "reference": ref, "type": "simple"})
        idx += 1

    # ── Multi-hop (175개) ─────────────────────────────────────
    multi_templates = [
        ("{p1} 교수와 {p2} 교수가 공동 담당하는 과목은?",
         "{shared}",
         "{p1}와 {p2} 담당 과목 교차"),
        ("{p1} 교수가 소속된 학과의 다른 교수는?",
         "{same_dept_prof}",
         "{p1} 소속 학과 → 동일 학과 교수"),
        ("{p1} 교수가 담당하는 과목을 수강하려면 어느 학과인가?",
         "{dept}",
         "{p1} 담당 과목 → 개설 학과"),
        ("{p1} 교수와 협력하는 교수의 담당 과목은?",
         "{collab_courses}",
         "{p1} 협력 교수 → 해당 교수 담당 과목"),
    ]

    collab = {
        "김철수": ["이영희"],
        "이영희": ["김철수","박민수"],
        "박민수": ["이영희"],
        "정수진": [],
    }
    shared_courses = {
        ("김철수","이영희"): "딥러닝",
        ("이영희","정수진"): "컴퓨터비전",
    }

    while len([d for d in dataset if d['type']=='multi_hop']) < 175:
        p1, p2 = random.sample(professors, 2)
        tmpl, ans_tmpl, ref_tmpl = random.choice(multi_templates)
        key = (p1['name'], p2['name'])
        rkey = (p2['name'], p1['name'])
        shared = shared_courses.get(key, shared_courses.get(rkey, "공동 담당 과목 없음"))
        same_dept = [p['name'] for p in professors
                     if p['dept'] == p1['dept'] and p['name'] != p1['name']]
        coll = collab.get(p1['name'], [])
        coll_courses = ", ".join(
            sum([p['courses'] for p in professors if p['name'] in coll], [])
        ) or "없음"

        q = tmpl.format(p1=p1['name'], p2=p2['name'])
        a = ans_tmpl.format(
            shared=shared,
            same_dept_prof=", ".join(same_dept) or "없음",
            dept=p1['dept'],
            collab_courses=coll_courses,
        )
        ref = ref_tmpl.format(p1=p1['name'], p2=p2['name'])
        dataset.append({"id": idx, "query": q, "answer": a,
                        "reference": ref, "type": "multi_hop"})
        idx += 1

    # ── Conditional (125개) ───────────────────────────────────
    cond_templates = [
        ("{dept} 소속 {age}세 이하 교수는?",
         lambda dept, age: ", ".join([p['name'] for p in professors
                                      if p['dept']==dept and p['age']<=age]) or "없음"),
        ("{age}세 이상 교수가 담당하는 과목은?",
         lambda dept, age: ", ".join(sum([p['courses'] for p in professors
                                          if p['age']>=age], [])) or "없음"),
        ("전임 교수 중 {age}세 미만인 사람은?",
         lambda dept, age: ", ".join([p['name'] for p in professors
                                      if p['age']<age]) or "없음"),
    ]

    while len([d for d in dataset if d['type']=='conditional']) < 125:
        dept = random.choice(list(depts.keys()))
        age  = random.choice([40, 45, 50, 55])
        tmpl, ans_fn = random.choice(cond_templates)
        q = tmpl.format(dept=dept, age=age)
        a = ans_fn(dept, age)
        ref = f"교수 나이/소속 조건 필터: {dept}, {age}세"
        dataset.append({"id": idx, "query": q, "answer": a,
                        "reference": ref, "type": "conditional"})
        idx += 1

    random.shuffle(dataset)
    for i, d in enumerate(dataset):
        d['id'] = i + 1
    return dataset


def save_dataset(path: str = "data/gold_qa_500.json"):
    """데이터셋 JSON 저장"""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ds = build_gold_dataset()
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(ds, f, ensure_ascii=False, indent=2)

    counts = {t: sum(1 for d in ds if d['type']==t)
              for t in ['simple','multi_hop','conditional']}
    print(f"✅ Gold QA 데이터셋 저장: {path}")
    print(f"   총 {len(ds)}개 | {counts}")
    return ds


if __name__ == "__main__":
    save_dataset()
