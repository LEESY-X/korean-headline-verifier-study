"""Build original paired Korean news cases without third-party source text."""

import argparse
import itertools
import json
from pathlib import Path


REGIONS = ("나온", "라온", "도림", "비솔", "하람", "여울", "다솜", "한결", "솔내", "미온")
TOPICS = (
    ("공동 장비 대여", "지역 교육관", "점검 시간표"),
    ("야간 자전거 정비", "교통지원실", "우천 시 운영 시간"),
    ("어린이 책 교환", "동네문화관", "보관 장소"),
    ("분리배출 안내", "생활환경실", "현장 점검 일정"),
    ("상점 전력 점검", "에너지지원실", "전력 기록 방식"),
    ("우천 대피 안내", "시설관리실", "모의 훈련 일정"),
    ("보행로 표지 정비", "도로지원관", "설치 위치"),
    ("주말 체육실 개방", "체육지원실", "예약 변경 절차"),
    ("노후 우산 수리", "자원순환센터", "접수 마감 시간"),
    ("이동 전시 운영", "전시지원팀", "반입 동선"),
    ("방과 후 악기 대여", "예술교육팀", "반납 점검 절차"),
    ("실내 작업실 예약", "주민협력관", "분기 운영 일정"),
)
RISKS = ("quantity", "certainty", "scope", "attribution")


def story(region, topic, risk, index):
    name, office, detail = topic
    project = f"{region} {name} 사업"
    organizer = f"{region} 운영위원회"
    partner = f"{region} {office}"
    lead = f"{project}과 관련한 안내가 이번 주 지역 회의에서 공유됐다."
    context = f"{partner}은 {detail}을 다음 안내문에 함께 담을 예정이다."

    if risk == "quantity":
        amount, waitlist = 21 + 3 * index, 17 + 2 * index
        decisive = f"올해 {project}의 최종 지원 규모는 {amount}곳이다."
        distractor = f"작년에는 예비 명단에만 {waitlist}곳이 이름을 올렸다."
        headlines = (f"{project}, 올해 {amount}곳 최종 지원", f"{project}, 올해 {waitlist}곳 최종 지원")
    elif risk == "certainty":
        decisive = f"{project}의 확대안은 심의 안건으로 올라갔지만 아직 의결되지 않았다."
        distractor = f"기존 운영 지침의 문구 수정은 앞선 회의에서 확정됐다."
        headlines = (f"{project} 확대안, 심의에 올라", f"{project} 확대안, 시행 확정")
    elif risk == "scope":
        cutoff = ("지난달 말", "지난 분기 말", "지난주 금요일")[index % 3]
        decisive = f"{project}의 신규 지원은 {cutoff}까지 지역에 등록한 단체에만 적용된다."
        distractor = f"그 이후에 등록한 단체도 설명회에 참석할 수 있지만 신규 지원 대상에는 포함되지 않는다."
        headlines = (f"{project}, 기한까지 등록한 단체만 신규 지원", f"{project}, 등록 시점과 무관하게 신규 지원")
    else:
        decisive = f"{organizer}가 {project}의 개편안을 공식 발표했다."
        distractor = f"{partner}은 회의 장소만 제공했으며 개편안을 발표하지 않았다."
        headlines = (f"{organizer}, {name} 개편안 발표", f"{partner}, {name} 개편안 발표")

    before = index % 3
    sentences = [lead, distractor, context]
    sentences.insert(before + 1, decisive)
    return sentences, before + 2, headlines


def make_cases(per_risk):
    combinations = list(itertools.product(REGIONS, TOPICS))
    for offset, risk in enumerate(RISKS):
        for index in range(per_risk):
            region, topic = combinations[(per_risk * offset + index) % len(combinations)]
            sentences, decisive, headlines = story(region, topic, risk, index)
            group = f"fiction-{risk}-{index + 1:03d}"
            for label, headline in zip(("supported", "misleading"), headlines, strict=True):
                yield dict(id=f"{group}-{label}", group=group, split="test", risk=risk, headline=headline,
                           sentences=sentences, gold=label, gold_evidence_id=decisive)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-risk", type=int, default=30)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.per_risk <= 30:
        parser.error("Choose 1 to 30 families per risk with unique setting and project combinations.")
    with args.out.open("x", encoding="utf-8", newline="\n") as output:
        for row in make_cases(args.per_risk):
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
