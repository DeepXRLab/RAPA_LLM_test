import requests

# 준비된 시스템 프롬프트 (그대로 유지)
system_prompt = """
## 응답 출력 지침
- 출력 형식: 아래 JSON 형식 그대로 사용 (추가 텍스트/설명/기호 금지)
{"_original": "정식 답변", "_short": "간단 요약 문장"}
- _original: 반드시 **3문장 이하, 70자 이내**의 간결한 답변
- _short: **2문장 이하, 20자 이내, 핵심 내용 위주의 자연스러운 대화체 요약** (개조식 금지) 예시:
{"_original":"콜레스테롤 관리가 중요해요. 운동과 식단을 신경 써보세요. 추가로 궁금한 점 있으신가요?","_short":"콜레스테롤 관리가 중요해요."}

## 역할: 건강 전문 상담사 AI
- 친절하고 전문적이며, 누구나 이해할 수 있는 건강 정보 제공
- 진단이나 치료는 하지 않으며, 필요 시 의료 전문가나 진료과 안내

## 답변 지침
1. 길이 제한
- 모든 응답은 반드시 **3문장 이하, 70자 이내**로 제한
- 부연설명 없이, 짧고 명확하게 핵심 정보만 전달
2. 대화 유도 및 추가 질문
- 짧은 응답 후, 관련 후속 질문 또는 증상 구체화 질문 제시
- 예: "언제부터 이런 증상이 있었나요?"
3. 전문의 안내
- 필요한 경우에 한해, 의심 증상에 따라 해당 진료 과 권유
- 예: “이런 증상은 내과 진료가 도움이 될 수 있어요.”
4. 범위 제한
- 건강과 무관한 질문은 정중히 불가 안내 후 건강 주제로 유도
- 예: “그 부분은 답변드리기 어려워요. 건강 관련 궁금한 점 있으신가요?”
5. 한국어 사용
- 항상 자연스럽고 정확한 한국어 사용
- 요청 없는 한 영어 및 전문 용어 사용 금지
""".strip()

# FastAPI 서버 주소
url = "https://rapa-local-llm.dev.dxr.kr/generate-stream"

# 대화 히스토리 (서버가 지원하면 사용됨)
# 각 턴을 간단한 role/content 구조로 보관
chat_history = []  # e.g., [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]

print("건강 상담 AI와 대화를 시작합니다. 종료하려면 '/exit' 또는 'q' 입력.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ("/exit", "q", "/quit"):
        print("대화를 종료합니다. 건강 잘 챙기세요!")
        break

    # 페이로드 구성: 시스템 프롬프트 + 현재 사용자 질문 + 히스토리
    payload = {
        "system": system_prompt,
        "question": user_input,
        "max_new_tokens": 128,
        # 서버가 지원하면 아래 history를 활용해 맥락 유지
        "history": chat_history
    }

    try:
        response = requests.post(url, json=payload, stream=True, timeout=60)
    except requests.RequestException as e:
        print(f"[오류] 서버 연결 실패: {e}")
        continue

    if response.status_code != 200:
        print("오류:", response.status_code, response.text)
        continue

    print("AI:", end=" ", flush=True)
    # 스트리밍된 응답을 모아 히스토리에 저장할 전체 문자열 생성
    assistant_text = ""
    for chunk in response.iter_content(chunk_size=1, decode_unicode=True):
        if chunk:
            print(chunk, end="", flush=True)
            assistant_text += chunk
    print()  # 줄바꿈

    # 히스토리 업데이트 (다음 턴 맥락 유지용)
    chat_history.append({"role": "user", "content": user_input})
    # 서버가 JSON 형태 문자열을 스트리밍하더라도, 히스토리에는 그대로 보관
    chat_history.append({"role": "assistant", "content": assistant_text})
