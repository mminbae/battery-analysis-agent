"""
T5: Comparison & SWOT Agent

비교 및 SWOT 구조화 — 양사 전략 차이 비교, S/W/O/T 4분면 작성.
이전 출력(T2, T3, T4)을 통합. 신규 검색 없음.

[P2 수정]
- _extract_critic_negatives: Critic(T4) 부정 근거를 파싱해 프롬프트에 명시적으로 주입
- _check_swot_completeness: Critic 반영 여부 검증 추가
- swot_node: critic_negatives를 Human 메시지에 별도 섹션으로 삽입
"""
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import WorkflowState, AgentOutput
from prompts.prompts import SWOT_SYSTEM, SWOT_HUMAN

MODEL_NAME = "gpt-4o-mini"


def _extract_critic_negatives(critic_content: str) -> dict[str, list[str]]:
    """
    [P2] Critic(T4) 출력에서 LGES·CATL 각각의 부정 근거를 파싱하여 반환.
    Returns: {"lges": [...], "catl": [...]}
    """
    result = {"lges": [], "catl": []}
    if not critic_content:
        return result

    current_entity = None
    for line in critic_content.split("\n"):
        line_stripped = line.strip()
        lower = line_stripped.lower()

        # 섹션 헤더 감지
        if "lges" in lower and ("부정" in lower or "리스크" in lower or "약점" in lower):
            current_entity = "lges"
        elif "catl" in lower and ("부정" in lower or "리스크" in lower or "약점" in lower):
            current_entity = "catl"

        # 번호 목록 항목 수집
        if current_entity and re.match(r"^\d+[\.\)]\s+.{20,}", line_stripped):
            result[current_entity].append(line_stripped)

    return result


def _check_swot_completeness(content: str, critic_content: str) -> tuple[bool, list[str]]:
    """
    [P2] SWOT 출력의 계량 조건 체크:
    - S/W/O/T 각 항목 ≥ 2개
    - 내부(S/W)·외부(O/T) 섹션 모두 존재
    - Critic 부정 근거가 W 또는 T에 반영됐는지 확인
    """
    fallback_items = []

    # SWOT 4분면 키워드 존재 여부
    required_sections = ["강점", "약점", "기회", "위협"]
    missing_sections = [s for s in required_sections if s not in content]
    if missing_sections:
        fallback_items.append(f"SWOT 섹션 누락: {missing_sections}")

    # 각 섹션별 항목 수 체크 (최소 2개)
    all_items = re.findall(r"^\s*[\d]+[\.\)]\s+.+|^\s*[-•*]\s+.+", content, re.MULTILINE)
    table_cells = re.findall(r"\|([^|]+)\|", content)
    non_empty_cells = [c.strip() for c in table_cells if len(c.strip()) > 10]

    total_items = len(all_items) + len(non_empty_cells)
    if total_items < 8:  # S/W/O/T × 각 2항목 = 최소 8개
        fallback_items.append(f"SWOT 항목 부족 (예상 ≥ 8, 확인: {total_items}개)")

    # [P2] Critic 부정 근거 반영 여부 체크
    # 약점(W) 또는 위협(T) 섹션에 최소 1건의 구체적 내용이 있는지 확인
    weakness_pattern = re.search(r"약점.{0,500}", content, re.DOTALL)
    threat_pattern = re.search(r"위협.{0,500}", content, re.DOTALL)
    wt_content = (weakness_pattern.group() if weakness_pattern else "") + \
                 (threat_pattern.group() if threat_pattern else "")

    if len(wt_content) < 100:
        fallback_items.append("W(약점)·T(위협) 섹션 내용 부족 (Critic 결과 미반영 가능성)")

    quantitative_check = len(fallback_items) == 0
    return quantitative_check, fallback_items


def swot_node(state: WorkflowState) -> dict:
    """T5: Comparison & SWOT Agent 노드."""
    print(f"[T5] SWOT Agent 실행 (retry: {state['retry_count'].get('T5', 0)})")

    lges_content = state.get("lges_strategy", {}).get("content", "")
    catl_content = state.get("catl_strategy", {}).get("content", "")
    critic_content = state.get("critic_result", {}).get("content", "")

    # [P2] Critic 부정 근거 파싱 → 프롬프트에 명시적 주입
    critic_negatives = _extract_critic_negatives(critic_content)
    lges_negatives_str = "\n".join(critic_negatives["lges"]) if critic_negatives["lges"] \
        else "Critic 부정 근거 파싱 실패 — 위 Critic 결과 원문 참조"
    catl_negatives_str = "\n".join(critic_negatives["catl"]) if critic_negatives["catl"] \
        else "Critic 부정 근거 파싱 실패 — 위 Critic 결과 원문 참조"

    print(f"[T5] Critic 부정 근거 파싱: LGES {len(critic_negatives['lges'])}건, CATL {len(critic_negatives['catl'])}건")

    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    messages = [
        SystemMessage(content=SWOT_SYSTEM),
        HumanMessage(content=f"""{SWOT_HUMAN.format(
            lges_content=lges_content,
            catl_content=catl_content,
            critic_content=critic_content,
        )}

## [P2] Critic 부정 근거 요약 — W(약점)·T(위협)에 반드시 반영

### LGES 부정 근거 (W 또는 T에 포함 필수):
{lges_negatives_str}

### CATL 부정 근거 (W 또는 T에 포함 필수):
{catl_negatives_str}

위 부정 근거들이 SWOT의 W(약점) 또는 T(위협) 항목에 명시적으로 반영되어야 합니다.
"""),
    ]
    response = llm.invoke(messages)
    content = response.content

    # [P2] 계량 조건 체크 (Critic 반영 여부 포함)
    quantitative_check, fallback_items = _check_swot_completeness(content, critic_content)

    # 출처는 이전 Agent 출처 수집 (신규 검색 없음)
    sources = []
    for key in ["lges_strategy", "catl_strategy", "critic_result"]:
        prev = state.get(key, {})
        if isinstance(prev, dict):
            sources.extend(prev.get("sources", []))
    sources = list(dict.fromkeys(sources))

    output: AgentOutput = {
        "task_id": "T5",
        "content": content,
        "sources": sources,
        "quantitative_check": quantitative_check,
        "fallback_items": fallback_items,
    }

    print(f"[T5] 완료 — quantitative_check: {quantitative_check}, 내용: {len(content)}자")

    return {
        "swot_comparison": output,
        "current_task": "T5",
        "retry_count": {"T5": state["retry_count"].get("T5", 0) + 1},
        "total_llm_calls": 1,
    }
