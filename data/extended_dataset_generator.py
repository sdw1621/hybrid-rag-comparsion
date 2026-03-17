"""
확장 Gold QA Dataset Generator
1,000개 질문-정답-참조근거 쌍
Simple 40%(400) / Multi-hop 35%(350) / Conditional 25%(250)
30명 교수 / 8개 학과 / 40개 과목 / 15개 프로젝트 기반
"""
import json, random, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.university_data import (
    DEPARTMENTS, PROFESSORS, COURSES, PROJECTS, COLLABORATIONS,
    get_prof_courses, get_dept_profs, get_proj_profs, get_prof_by_id
)


def _courses_of(prof_id):
    return [c for c in COURSES if prof_id in c["profs"]]

def _dept_profs(dept_name):
    return [p for p in PROFESSORS if p["dept"] == dept_name]

def _proj_of(prof_id):
    return [pr for pr in PROJECTS if prof_id in pr["profs"]]


def build_extended_dataset(seed: int = 42) -> list:
    random.seed(seed)
    dataset = []

    # ────────────────────────────────────────────────────────
    # SIMPLE 400개 — 단일 속성/관계 질의
    # ────────────────────────────────────────────────────────
    simple_templates = [
        # 교수 → 소속
        (lambda: (p := random.choice(PROFESSORS),
                  f"{p['name']} 교수의 소속 학과는?",
                  p['dept'],
                  f"{p['name']} 교수 소속 정보",
                  "prof_dept")),
        # 교수 → 나이
        (lambda: (p := random.choice(PROFESSORS),
                  f"{p['name']} 교수의 나이는?",
                  f"{p['age']}세",
                  f"{p['name']} 교수 나이 정보",
                  "prof_age")),
        # 교수 → 직급
        (lambda: (p := random.choice(PROFESSORS),
                  f"{p['name']} 교수의 직급은?",
                  p['rank'],
                  f"{p['name']} 교수 직급 정보",
                  "prof_rank")),
        # 교수 → 이메일
        (lambda: (p := random.choice(PROFESSORS),
                  f"{p['name']} 교수의 이메일 주소는?",
                  p['email'],
                  f"{p['name']} 교수 연락처",
                  "prof_email")),
        # 교수 → 연구분야
        (lambda: (p := random.choice(PROFESSORS),
                  f"{p['name']} 교수의 연구 분야는?",
                  ", ".join(p['research']),
                  f"{p['name']} 교수 연구 정보",
                  "prof_research")),
        # 교수 → 담당 과목
        (lambda: (p := random.choice(PROFESSORS),
                  f"{p['name']} 교수가 담당하는 과목은?",
                  ", ".join(c['name'] for c in _courses_of(p['id'])) or "없음",
                  f"{p['name']} 교수 담당 과목",
                  "prof_courses")),
        # 과목 → 담당 교수
        (lambda: (c := random.choice(COURSES),
                  f"{c['name']} 과목을 담당하는 교수는?",
                  ", ".join(get_prof_by_id(pid)['name'] for pid in c['profs'] if get_prof_by_id(pid)),
                  f"{c['name']} 과목 담당 정보",
                  "course_prof")),
        # 과목 → 학점
        (lambda: (c := random.choice(COURSES),
                  f"{c['name']} 과목은 몇 학점인가?",
                  f"{c['credits']}학점",
                  f"{c['name']} 과목 학점 정보",
                  "course_credit")),
        # 과목 → 개설 학과
        (lambda: (c := random.choice(COURSES),
                  f"{c['name']} 과목은 어느 학과에서 개설하는가?",
                  c['dept'],
                  f"{c['name']} 개설 학과",
                  "course_dept")),
        # 학과 → 소속 교수 목록
        (lambda: (d := random.choice(DEPARTMENTS),
                  f"{d['name']}에 소속된 교수는 누구인가?",
                  ", ".join(p['name'] for p in _dept_profs(d['name'])) or "없음",
                  f"{d['name']} 소속 교수 목록",
                  "dept_profs")),
        # 학과 → 단과대학
        (lambda: (d := random.choice(DEPARTMENTS),
                  f"{d['name']}은 어느 단과대학 소속인가?",
                  d['college'],
                  f"{d['name']} 소속 단과대학",
                  "dept_college")),
        # 프로젝트 → 참여 교수
        (lambda: (pr := random.choice(PROJECTS),
                  f"{pr['name']} 프로젝트에 참여하는 교수는?",
                  ", ".join(get_prof_by_id(pid)['name'] for pid in pr['profs'] if get_prof_by_id(pid)),
                  f"{pr['name']} 프로젝트 참여 교수",
                  "proj_profs")),
        # 프로젝트 → 연구 분야
        (lambda: (pr := random.choice(PROJECTS),
                  f"{pr['name']} 프로젝트의 연구 분야는?",
                  pr['field'],
                  f"{pr['name']} 프로젝트 분야",
                  "proj_field")),
    ]

    idx = 1
    attempts = 0
    while len(dataset) < 400 and attempts < 5000:
        attempts += 1
        try:
            fn = random.choice(simple_templates)
            result = fn()
            _, q, a, ref, qtype = result
            if not a or a == "없음":
                continue
            dataset.append({"id": idx, "query": q, "answer": a,
                             "reference": ref, "type": "simple"})
            idx += 1
        except Exception:
            continue

    # ────────────────────────────────────────────────────────
    # MULTI-HOP 350개 — 2단계 이상 관계 탐색
    # ────────────────────────────────────────────────────────
    dept_map = {d["name"]: d for d in DEPARTMENTS}

    def mh_cases():
        cases = []
        for p1 in PROFESSORS:
            # 같은 학과 다른 교수
            same = [p for p in PROFESSORS if p["dept"]==p1["dept"] and p["id"]!=p1["id"]]
            if same:
                p2 = random.choice(same)
                cases.append((
                    f"{p1['name']} 교수와 같은 학과 소속인 다른 교수는?",
                    ", ".join(p["name"] for p in same),
                    f"{p1['name']} 소속 학과 → 동일 학과 교수 목록"
                ))
            # 협력 교수의 담당 과목
            collabs = [p2 for p1id, p2id in COLLABORATIONS
                       if p1id == p1["id"]
                       for p2 in PROFESSORS if p2["id"] == p2id]
            collabs += [p1_ for p2id, p1id in COLLABORATIONS
                        if p1id == p1["id"]
                        for p1_ in PROFESSORS if p1_["id"] == p2id]
            if collabs:
                cp = random.choice(collabs)
                cp_courses = _courses_of(cp["id"])
                if cp_courses:
                    cases.append((
                        f"{p1['name']} 교수와 협력하는 교수의 담당 과목은?",
                        ", ".join(c["name"] for c in cp_courses),
                        f"{p1['name']} 협력 교수 → 담당 과목"
                    ))
            # 교수의 학과 단과대학
            d = dept_map.get(p1["dept"])
            if d:
                cases.append((
                    f"{p1['name']} 교수가 소속된 학과의 단과대학은?",
                    d["college"],
                    f"{p1['name']} → 학과 → 단과대학"
                ))
            # 프로젝트 참여 → 프로젝트의 다른 참여 교수
            projs = _proj_of(p1["id"])
            if projs:
                pr = random.choice(projs)
                others = [get_prof_by_id(pid)["name"]
                          for pid in pr["profs"]
                          if pid != p1["id"] and get_prof_by_id(pid)]
                if others:
                    cases.append((
                        f"{p1['name']} 교수가 참여하는 프로젝트의 다른 구성원은?",
                        ", ".join(others),
                        f"{p1['name']} → 프로젝트 → 공동 참여 교수"
                    ))
        return cases

    mh_pool = mh_cases()
    random.shuffle(mh_pool)
    for q, a, ref in mh_pool:
        if len(dataset) - 400 >= 350:
            break
        if not a:
            continue
        dataset.append({"id": idx, "query": q, "answer": a,
                         "reference": ref, "type": "multi_hop"})
        idx += 1
    # 부족하면 반복 샘플
    while len(dataset) - 400 < 350:
        q, a, ref = random.choice(mh_pool)
        if a:
            dataset.append({"id": idx, "query": q, "answer": a,
                             "reference": ref, "type": "multi_hop"})
            idx += 1

    # ────────────────────────────────────────────────────────
    # CONDITIONAL 250개 — 수치 조건 + 논리 제약
    # ────────────────────────────────────────────────────────
    age_thresholds = [35, 38, 40, 42, 45, 48, 50, 52]
    ops = [
        ("이하",  lambda a, t: a <= t),
        ("미만",  lambda a, t: a <  t),
        ("이상",  lambda a, t: a >= t),
        ("초과",  lambda a, t: a >  t),
    ]
    ranks = ["정교수","부교수","조교수"]

    cond_pool = []
    for dept in DEPARTMENTS:
        dp = _dept_profs(dept["name"])
        for thr in age_thresholds:
            for op_str, op_fn in ops:
                filtered = [p["name"] for p in dp if op_fn(p["age"], thr)]
                if filtered:
                    cond_pool.append((
                        f"{dept['name']} 소속 {thr}세 {op_str} 교수는?",
                        ", ".join(filtered),
                        f"{dept['name']} 소속 교수 나이 조건 필터 ({thr}세 {op_str})"
                    ))

    for thr in age_thresholds:
        for op_str, op_fn in ops:
            filtered = [p["name"] for p in PROFESSORS if op_fn(p["age"], thr)]
            if filtered:
                cond_pool.append((
                    f"전체 교수 중 {thr}세 {op_str}인 사람은?",
                    ", ".join(filtered),
                    f"전체 교수 나이 조건 필터 ({thr}세 {op_str})"
                ))

    for rank in ranks:
        filtered = [p["name"] for p in PROFESSORS if p["rank"] == rank]
        cond_pool.append((
            f"{rank}에 해당하는 교수 목록은?",
            ", ".join(filtered),
            f"직급 = {rank} 필터"
        ))
        for dept in DEPARTMENTS:
            dp_rank = [p["name"] for p in PROFESSORS
                       if p["dept"]==dept["name"] and p["rank"]==rank]
            if dp_rank:
                cond_pool.append((
                    f"{dept['name']} 소속 {rank}은 누구인가?",
                    ", ".join(dp_rank),
                    f"{dept['name']} + 직급={rank} 복합 필터"
                ))

    random.shuffle(cond_pool)
    for q, a, ref in cond_pool:
        if len(dataset) - 750 >= 250:
            break
        dataset.append({"id": idx, "query": q, "answer": a,
                         "reference": ref, "type": "conditional"})
        idx += 1
    while len(dataset) - 750 < 250:
        q, a, ref = random.choice(cond_pool)
        dataset.append({"id": idx, "query": q, "answer": a,
                         "reference": ref, "type": "conditional"})
        idx += 1

    # ── 셔플 + ID 재부여 ──────────────────────────────────
    random.shuffle(dataset)
    for i, d in enumerate(dataset):
        d["id"] = i + 1

    return dataset


def save_extended_dataset(path: str = "data/gold_qa_1000.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ds = build_extended_dataset()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ds, f, ensure_ascii=False, indent=2)

    counts = {t: sum(1 for d in ds if d["type"] == t)
              for t in ["simple", "multi_hop", "conditional"]}
    print(f"✅ 확장 Gold QA 저장: {path}")
    print(f"   총 {len(ds)}개 | {counts}")
    print(f"   비율: Simple {counts['simple']/len(ds)*100:.1f}%"
          f" / Multi-hop {counts['multi_hop']/len(ds)*100:.1f}%"
          f" / Conditional {counts['conditional']/len(ds)*100:.1f}%")
    return ds


if __name__ == "__main__":
    save_extended_dataset()
