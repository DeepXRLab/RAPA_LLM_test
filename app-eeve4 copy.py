from fastapi import FastAPI
from flask import Response
import re, json
from pydantic import BaseModel
from typing import List, Literal, Optional
import torch, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
from fastapi.responses import StreamingResponse
import time, os, threading

app = FastAPI()

MODEL_REPO = "kkh27/healthcareLLM_v4"
MODEL_DIR = "./model/healthcareLLM_v4"

print(torch.__version__)
print(transformers.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())

# === 모델/토크나이저 로딩 ===
if not os.path.exists(MODEL_DIR) or not os.listdir(MODEL_DIR):
    print(f"모델을 {MODEL_REPO}에서 {MODEL_DIR}로 다운로드합니다...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    model = AutoModelForCausalLM.from_pretrained(MODEL_REPO, torch_dtype="auto")
    tokenizer.save_pretrained(MODEL_DIR)
    model.save_pretrained(MODEL_DIR)
else:
    print(f"모델을 {MODEL_DIR}에서 불러옵니다...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype="auto")

# 단일 GPU 명시 (환경에 맞게 조정)
model.to("cuda:1")
print("모델 디바이스:", next(model.parameters()).device)

# === 요청 스키마: history 추가 ===
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class PromptRequest(BaseModel):
    system: str
    question: str
    max_new_tokens: int = 128
    history: Optional[List[ChatMessage]] = None

# === 프롬프트 유틸 ===
def apply_template(system: str, history: Optional[List[ChatMessage]], question: str) -> str:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    if history:
        # 안전하게 role/content만 반영
        for m in history:
            msgs.append({"role": m.role, "content": m.content})
    msgs.append({"role": "user", "content": question})

    # 1) chat_template 있으면 사용
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # 2) 수동 포맷 (Human/Assistant)
        parts = [
            "A chat between a curious user and an artificial intelligence assistant.",
            "The assistant gives helpful, detailed, and polite answers to the user's questions."
        ]
        if system:
            parts.append(f"System: {system.strip()}")
        if history:
            for m in history:
                prefix = {"user": "Human", "assistant": "Assistant", "system": "System"}[m.role]
                parts.append(f"{prefix}: {m.content.strip()}")
        parts.append(f"Human: {question.strip()}")
        parts.append("Assistant:")
        return "\n".join(parts)

def count_tokens(text: str) -> int:
    return tokenizer(text, return_tensors="pt").input_ids.shape[-1]

def window_history(system: str, history: Optional[List[ChatMessage]], question: str, max_new_tokens: int):
    model_max = getattr(model.config, "max_position_embeddings",
                        getattr(tokenizer, "model_max_length", 4096))
    reserve = max_new_tokens + 50  # 여유 버퍼
    kept: List[ChatMessage] = []
    if history:
        # 최근부터 거꾸로 쌓아가며 한도 체크
        for m in reversed(history):
            kept.insert(0, m)
            prompt = apply_template(system, kept, question)
            if count_tokens(prompt) + reserve > model_max:
                kept.pop(0)
                break
    prompt = apply_template(system, kept, question)
    return kept, prompt

# (옵션) “다음 턴 Human:”이 생성되면 중단시키는 스톱 조건
class StopOnHumanPrefix(StoppingCriteria):
    def __init__(self, stop_strings: List[str], tokenizer):
        super().__init__()
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
        self.cache = ""

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        # 최신 토큰을 텍스트로 변환(간단/비효율적이지만 짧은 max_new_tokens라면 실용적)
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        # 캐시와 합쳐 검사
        self.cache = text[-2000:]  # 메모리 과다 방지
        return any(s in self.cache for s in self.stop_strings)

def build_generation_kwargs(input_ids, max_new_tokens):
    stop_criteria = StoppingCriteriaList([
        StopOnHumanPrefix(stop_strings=["\nHuman:", "\nUser:"], tokenizer=tokenizer)
    ])
    return dict(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=1e-5,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        stopping_criteria=stop_criteria
    )

# === 논스트리밍 ===
@app.post("/generate")
@torch.inference_mode()
def generate(req: PromptRequest):
    t0 = time.time()
    print("== [START] == 프롬프트 수신 및 인퍼런스 시작")

    kept, prompt = window_history(req.system, req.history, req.question, req.max_new_tokens)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    print("입력 토큰 길이:", input_ids.shape[-1])

    output = model.generate(**build_generation_kwargs(input_ids, req.max_new_tokens))
    # 입력 길이 이후만 디코딩
    result = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

    t1 = time.time()
    print(f"== [DONE] == 걸린 시간: {t1-t0:.2f}초\n")
    return {"response": result}

# === 스트리밍 ===
@app.post("/generate-stream")
def generate_stream(req: PromptRequest):
    def stream_generator():
        print("== [START] == 프롬프트 수신 및 인퍼런스 시작")
        kept, prompt = window_history(req.system, req.history, req.question, req.max_new_tokens)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = build_generation_kwargs(input_ids, req.max_new_tokens)
        gen_kwargs["streamer"] = streamer

        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text

    return StreamingResponse(stream_generator(), media_type="text/plain")


def extract_first_json(text: str):
    """
    텍스트에서 첫 번째 JSON 오브젝트만 추출
    """
    # { ... } 잡아오기 (가장 처음 매칭되는 부분)
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"_original": match.group(0)}  # 파싱 실패하면 문자열로 반환
    return {"_original": text}  # JSON 전혀 없으면 전체 문자열 반환



# === 시스템 프롬프트 강제 반영용 ===
@app.post("/generate-with-systemPromt")
@torch.inference_mode()
def generate_with_system_promt(req: PromptRequest):
    """
    요청에서 받은 system 프롬프트를 그대로 사용하여 응답 생성
    """
    t0 = time.time()
    print("== [START] == generate-with-systemPromt 호출")

    # system 프롬프트 강제 적용
    system_prompt = req.system if req.system else ""
    #print(system_prompt)
    
    limited_history = req.history[-10:] if req.history else []
    
    kept, prompt = window_history(system_prompt, limited_history, req.question, req.max_new_tokens)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    print("입력 토큰 길이:", input_ids.shape[-1])

    output = model.generate(**build_generation_kwargs(input_ids, req.max_new_tokens))
    result = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


    clean_result = extract_first_json(result)
    try:
        # 작은따옴표 dict → 큰따옴표 JSON으로 치환
        clean_result = clean_result.replace("'", '"')
        json.loads(clean_result)  # 유효성 검사
    except Exception:
        # 유효하지 않으면 그냥 문자열로 감싸서 내려줌
        clean_result = json.dumps({"_original": result}, ensure_ascii=False)
        
    t1 = time.time()
    print(f"== [DONE] == 걸린 시간: {t1-t0:.2f}초\n")
    print("답변:", clean_result)
    
    return {
        "response": clean_result
    }
    
    '''
    return Response(
        json.dumps({"response": json.dumps(clean_result, ensure_ascii=False)}, ensure_ascii=False),
        content_type="application/json; charset=utf-8"
    )
    '''




# 실행 (개발 테스트용)
# 터미널에서: uvicorn app-eeve4:app --host 0.0.0.0 --port 57777 --workers 1
