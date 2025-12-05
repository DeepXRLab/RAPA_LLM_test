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
import requests

import random, numpy as np, torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -----------------------
# FastAPI 앱 초기화
# -----------------------
app = FastAPI()

MODEL_REPO = "kkh27/healthcareLLM_v4_Tri-7B"
MODEL_DIR = "./model/healthcareLLM_v4_Tri-7B"
RAG_ENDPOINT = os.environ.get("RAG_ENDPOINT", "http://192.168.0.11:7774/")
RAG_TIMEOUT = float(os.environ.get("RAG_TIMEOUT", "4.0"))  # 초
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "5"))

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
    max_new_tokens: int = 32  # 기본 제한
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
        top_p=1.0, top_k=0, 
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
        stopping_criteria=stop_criteria,
        repetition_penalty=1.03,
        
        # ⬇️ 확률 계산을 위해 추가
        return_dict_in_generate=True,
        output_scores=True,
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



def call_rag_server(user_question: str) -> Optional[str]:
    """
    RAG Flask 서버의 '/' 엔드포인트에 POST하여 rag_result(str)를 받아온다.
    실패 시 None 반환.
    """
    try:
        # RAG 서버는 full_query에서 토큰을 추출하지만,
        # 토큰이 없어도 전체를 질의로 사용하므로 간단히 question만 보낸다.
        payload = {"query": user_question}
        resp = requests.post(
            RAG_ENDPOINT.rstrip("/") + "/",
            json=payload,
            timeout=RAG_TIMEOUT,
        )
        if resp.status_code == 200:
            data = resp.json()
            # 예시 RAG 응답: {"rag_result": "..."} 형태
            rag_text = data.get("rag_result")
            if isinstance(rag_text, str) and rag_text.strip():
                return rag_text
        else:
            print(f"[WARN] RAG HTTP {resp.status_code}: {resp.text[:300]}")
    except Exception as e:
        print(f"[WARN] RAG call failed: {e}")
    return None


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

def _best_effort_parse_inner_json(s: str) -> dict:
    """
    문자열 s 안의 JSON을 최대한 복구해 dict로 반환.
    실패하면 {"_original": s}.
    _short/_original 추출은 닫힘 따옴표/중괄호가 없어도 시도함.
    """
    if not isinstance(s, str):
        return {"_original": s}

    # 1) 정상 파싱 시도
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    fixed = s

    # 2) 흔한 깨짐 보정: 중괄호/콤마/따옴표 약간 정리
    open_braces = fixed.count("{")
    close_braces = fixed.count("}")
    if close_braces < open_braces:
        fixed = fixed + ("}" * (open_braces - close_braces))
    fixed = re.sub(r',\s*}', '}', fixed)

    # 2-1) 다시 파싱 시도
    try:
        parsed = json.loads(fixed)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # 3) 최후: 정규식으로 _original/_short 긁어오기 (닫힘 따옴표 없어도 허용)
    def _grab_relaxed(key: str):
        # 시작은 "_short": " 로 강제하되, 끝은 (" 로 닫히거나 문자열 끝/중괄호 전까지 허용)
        # 그룹은 큰따옴표 내부의 이스케이프 처리된 문자들 또는 (닫힘 따옴표가 없다면) 남은 전부.
        m = re.search(
            rf'"{key}"\s*:\s*"(?P<val>(?:[^"\\]|\\.)*)',  # 닫힘 " 없어도 매치됨
            s,
            flags=re.DOTALL,
        )
        if not m:
            return None
        raw = m.group('val')
        # 끝쪽에 붙은 말미 찌꺼기 정리: 따옴표/중괄호/공백/작은따옴표
        raw = re.sub(r'[\s\'"}]*$', '', raw)
        # 가능한 경우 JSON 문자열 언이스케이프 시도
        try:
            return json.loads(f'"{raw}"')
        except Exception:
            return raw

    out = {}
    o = _grab_relaxed("_original")
    sh = _grab_relaxed("_short")
    if o is not None:
        out["_original"] = o
    if sh is not None:
        out["_short"] = sh

    if out:
        return out

    # 그래도 실패하면 원문 통째로
    return {"_original": s}


def normalize_output(obj: dict) -> dict:
    """
    모델 출력을 {"_original": "...", "_short": (str|None)} 포맷으로 강제.
    - _original/_short 가 JSON 문자열로 중첩돼 있어도 끝까지 풀어냄
    - 내부 값이 있으면 내부(inner) 우선
    """
    if not isinstance(obj, dict):
        return {"_original": str(obj), "_short": None}

    result = {"_original": "", "_short": None}

    # 1) 바깥 _original 후보 결정
    inner_candidate = obj.get("_original", obj)

    # 2) 문자열이면 베스트-에포트 파싱, dict면 그대로
    if isinstance(inner_candidate, str):
        inner_dict = _best_effort_parse_inner_json(inner_candidate)
    elif isinstance(inner_candidate, dict):
        inner_dict = inner_candidate
    else:
        inner_dict = {"_original": inner_candidate}

    # 3) 최종 _original
    inner_orig = inner_dict.get("_original", inner_dict)
    if isinstance(inner_orig, (dict, list)):
        result["_original"] = json.dumps(inner_orig, ensure_ascii=False)
    else:
        result["_original"] = str(inner_orig)

    # 4) 최종 _short (안쪽 > 바깥 우선)
    inner_short = inner_dict.get("_short", None)
    if inner_short is not None:
        result["_short"] = str(inner_short)
    else:
        outer_short = obj.get("_short", None)
        result["_short"] = str(outer_short) if outer_short is not None else None

    return result

@app.post("/generate-with-systemPromt")
@torch.inference_mode()
def generate_with_system_promt(req: PromptRequest):
    import math
    import torch.nn.functional as F

    t0 = time.time()

    system_prompt = req.system if req.system else ""
    limited_history = req.history[-10:] if req.history else []
    kept, prompt = window_history(system_prompt, req.history, req.question, req.max_new_tokens)
    print("질문: " + req.question)

    tokens = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = tokens.input_ids.to(model.device)
    attention_mask = tokens.attention_mask.to(model.device)

    print("입력 토큰 길이:", input_ids.shape[-1])

    # ---------- 1차 생성 & 확률 ----------
    gen_out = model.generate(**build_generation_kwargs(input_ids, attention_mask, req.max_new_tokens))

    sequences = gen_out.sequences
    prompt_len = input_ids.shape[1]
    gen_ids = sequences[:, prompt_len:]

    chosen_token_probs = []
    if hasattr(gen_out, "scores") and gen_out.scores is not None and len(gen_out.scores) > 0:
        for step, logits in enumerate(gen_out.scores):
            probs = F.softmax(logits, dim=-1)
            step_token_ids = gen_ids[:, step]
            step_token_probs = probs.gather(1, step_token_ids.unsqueeze(1)).squeeze(1)
            chosen_token_probs.append(step_token_probs)
        chosen_token_probs = torch.stack(chosen_token_probs, dim=1)
    else:
        chosen_token_probs = torch.empty((input_ids.size(0), 0), device=input_ids.device)

    if chosen_token_probs.numel() > 0:
        eos_id = tokenizer.eos_token_id
        eos_mask = (gen_ids != eos_id).to(chosen_token_probs.dtype)
        lengths = eos_mask.sum(dim=1).clamp_min(1)
        masked_probs = chosen_token_probs * eos_mask
        avg_token_prob = (masked_probs.sum(dim=1) / lengths).mean().item()

        avg_nll = (-(masked_probs.clamp_min(1e-12).log().sum(dim=1) / lengths)).mean().item()
        perplexity = math.exp(avg_nll)
    else:
        avg_token_prob, avg_nll, perplexity = 0.0, float("inf"), float("inf")

    raw_result_1st = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()

    THRESHOLD = 0.99
    need_rag = (avg_token_prob <= THRESHOLD)

    t_mid = time.time()
    #print(f"[1st] gen_len={gen_ids.shape[1]}, time={t_mid - t0:.2f}s, avg_prob={avg_token_prob:.4f}, ppl={perplexity:.2f}")
    #print("== [1st DONE] ==")
    #print("초안 답변:", raw_result_1st)

    rag_used = False
    rag_text = None
    final_raw = raw_result_1st

    # ---------- RAG 경로 ----------
    if need_rag:
        #print("[RAG] need_rag=True → RAG 검색 진행")
        rag_text = call_rag_server(req.question)

        if rag_text:
            rag_used = True
            # 시스템 프롬프트 확장: 아래 블록을 system 밑에 추가
            augmented_system = (
                (system_prompt + "\n\n") if system_prompt else ""
            ) + (
                "-----\n"
                "### Retrieved Knowledge (RAG)\n"
                "(아래는 관련 지식 검색 결과입니다. 다음 내용은 질문과 관련있는 범위에서만 최우선 신뢰하여 사용하세요.)\n\n"
                f"{rag_text}\n"
                "-----\n"
            )

            # 확장된 시스템 프롬프트로 윈도우/프롬프트 재구성
            kept2, prompt2 = window_history(augmented_system, req.history, req.question, req.max_new_tokens)

            tokens2 = tokenizer(prompt2, return_tensors="pt", padding=True)
            input_ids2 = tokens2.input_ids.to(model.device)
            attention_mask2 = tokens2.attention_mask.to(model.device)

            #print("[RAG] 재생성용 입력 토큰 길이:", input_ids2.shape[-1])
            print("[RAG] RAG 적용 시스템 프롬프트:", augmented_system)

            gen_out2 = model.generate(**build_generation_kwargs(input_ids2, attention_mask2, req.max_new_tokens))
            sequences2 = gen_out2.sequences
            gen_ids2 = sequences2[:, input_ids2.shape[1]:]
            final_raw = tokenizer.decode(gen_ids2[0], skip_special_tokens=True).strip()

            #print("[RAG] 재생성 완료")
        else:
            print("[RAG] 검색 실패 또는 결과 없음 → 1차 답변 유지")

    # ---------- JSON 추출 & 응답 ----------
    clean_result = extract_first_json(final_raw)
    print("중간답변: ")
    print(clean_result)
    clean_result = normalize_output(clean_result)
    clean_result_str = json.dumps(clean_result, ensure_ascii=False)
    print("최종답변: " + clean_result_str)

    t1 = time.time()
    print(f"== [DONE] == 총 시간: {t1 - t0:.2f}s | RAG 사용: {rag_used}")

    # Go 서버 호환: 기존 "response"는 string(JSON string) 유지 + 메타 필드 추가
    return {"response": clean_result_str}
    
    
def generate_with_system_promt_old(req: PromptRequest):
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
# uvicorn app-Tri7B:app --host 0.0.0.0 --port 57777 --workers 1
