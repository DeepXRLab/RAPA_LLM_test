from fastapi import FastAPI
import re, json, time, os, threading
from pydantic import BaseModel
from typing import List, Literal, Optional
import torch, transformers
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
)
from fastapi.responses import StreamingResponse

# -----------------------
# FastAPI 앱 초기화
# -----------------------
app = FastAPI()

MODEL_REPO = "kkh27/healthcareLLM_v4_Tri-7B"
MODEL_DIR = "./model/healthcareLLM_v4_Tri-7B"

print("Torch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

# -----------------------
# 모델 / 토크나이저 로딩
# -----------------------
if not os.path.exists(MODEL_DIR) or not os.listdir(MODEL_DIR):
    print(f"모델을 {MODEL_REPO}에서 {MODEL_DIR}로 다운로드합니다...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa"
    )
    tokenizer.save_pretrained(MODEL_DIR)
    model.save_pretrained(MODEL_DIR)
else:
    print(f"모델을 {MODEL_DIR}에서 불러옵니다...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa"
    )

# 단일 GPU 고정
model.to("cuda:1")
print("모델 디바이스:", next(model.parameters()).device)

# -----------------------
# 요청 스키마
# -----------------------
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class PromptRequest(BaseModel):
    system: str
    question: str
    max_new_tokens: int = 96  # 기본 제한
    history: Optional[List[ChatMessage]] = None

# -----------------------
# 프롬프트 처리 유틸
# -----------------------
def apply_template(system: str, history: Optional[List[ChatMessage]], question: str) -> str:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    if history:
        for m in history:
            msgs.append({"role": m.role, "content": m.content})
    msgs.append({"role": "user", "content": question})

    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        parts = [
            "A chat between a curious user and an AI assistant.",
            "The assistant gives helpful, detailed, and polite answers."
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
    reserve = max_new_tokens + 50
    kept: List[ChatMessage] = []
    if history:
        for m in reversed(history):
            kept.insert(0, m)
            prompt = apply_template(system, kept, question)
            if count_tokens(prompt) + reserve > model_max:
                kept.pop(0)
                break
    prompt = apply_template(system, kept, question)
    return kept, prompt

# -----------------------
# StoppingCriteria
# -----------------------
STOP_STRINGS = [
    "\nHuman:", "\nUser:",
    "\n| user", "\n| assistant",
    "\nassistant:", "\nAssistant:",
    "}"  # JSON 닫힘
]

class StopOnHumanPrefix(StoppingCriteria):
    def __init__(self, stop_strings: List[str], tokenizer):
        super().__init__()
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
        self.prev_len = None
        self.cache = ""

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        if self.prev_len is None:
            self.prev_len = input_ids.shape[1]
            return False
        new_tokens = input_ids[0, self.prev_len:]
        if new_tokens.numel() > 0:
            self.cache += self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            self.cache = self.cache[-2000:]
            self.prev_len = input_ids.shape[1]
        return any(s in self.cache for s in self.stop_strings)

# -----------------------
# Generate kwargs builder
# -----------------------
def build_generation_kwargs(input_ids, attention_mask, max_new_tokens):
    max_new_tokens = min(max_new_tokens, 128)  # 강제 상한
    stop_criteria = StoppingCriteriaList([
        StopOnHumanPrefix(STOP_STRINGS, tokenizer)
    ])
    return dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
        stopping_criteria=stop_criteria,
        repetition_penalty=1.03
    )

# -----------------------
# JSON 추출 유틸
# -----------------------
def extract_first_json(text: str):
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"_original": match.group(0)}
    return {"_original": text}

# -----------------------
# API 엔드포인트
# -----------------------
@app.post("/generate")
@torch.inference_mode()
def generate(req: PromptRequest):
    t0 = time.time()
    kept, prompt = window_history(req.system, req.history, req.question, req.max_new_tokens)

    tokens = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = tokens.input_ids.to(model.device)
    attention_mask = tokens.attention_mask.to(model.device)

    print("== [START] == /generate 호출")
    print("입력 토큰 길이:", input_ids.shape[-1])

    output = model.generate(**build_generation_kwargs(input_ids, attention_mask, req.max_new_tokens))
    result = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

    t1 = time.time()
    print(f"== [DONE] == 걸린 시간: {t1-t0:.2f}초\n")
    return {"response": result}

@app.post("/generate-with-systemPromt")
@torch.inference_mode()
def generate_with_system_promt(req: PromptRequest):
    t0 = time.time()
    #print("== [START] == generate-with-systemPromt 호출")

    system_prompt = req.system if req.system else ""
    limited_history = req.history[-10:] if req.history else []
    kept, prompt = window_history(system_prompt, req.history, req.question, req.max_new_tokens)
    #print(system_prompt)
    
    tokens = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = tokens.input_ids.to(model.device)
    attention_mask = tokens.attention_mask.to(model.device)

    print("입력 토큰 길이:", input_ids.shape[-1])

    output = model.generate(**build_generation_kwargs(input_ids, attention_mask, req.max_new_tokens))
    raw_result = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

    clean_result = extract_first_json(raw_result)
    clean_result = json.dumps(clean_result, ensure_ascii=False)

    t1 = time.time()
    print(f"gen_len={output.shape[1]-input_ids.shape[1]}, time={t1-t0:.2f}s")
    print("== [DONE] == 걸린 시간: %.2fs\n" % (t1-t0))
    print("답변:", raw_result)

    # Go 서버에서 string으로 파싱 가능하도록 string으로 리턴
    return {"response": clean_result}

@app.post("/generate-stream")
def generate_stream(req: PromptRequest):
    def stream_generator():
        kept, prompt = window_history(req.system, req.history, req.question, req.max_new_tokens)

        tokens = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = tokens.input_ids.to(model.device)
        attention_mask = tokens.attention_mask.to(model.device)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = build_generation_kwargs(input_ids, attention_mask, req.max_new_tokens)
        gen_kwargs["streamer"] = streamer

        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text

    return StreamingResponse(stream_generator(), media_type="text/plain")


# 실행 예:
# uvicorn app-Tri7B:app --host 0.0.0.0 --port 57776 --workers 1
