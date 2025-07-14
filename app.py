from fastapi import FastAPI
from pydantic import BaseModel
import torch, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import time
import os

print(torch.__version__)
print(transformers.__version__)
print(torch.cuda.is_available())  # True 나오면 성공!
print(torch.cuda.device_count())  # 사용 가능한 GPU 개수

# FastAPI 앱
app = FastAPI()

# 모델/토크나이저 로딩 (서버 시작 시 메모리에 적재)
print("모델 로딩 중...")
#tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="left")
#base_model = AutoModelForCausalLM.from_pretrained(
#    BASE_MODEL,
#    torch_dtype=torch.bfloat16,        # A100에서 bfloat16 지원
#    device_map="auto"                  # 자동으로 GPU에 올림
#)
# LoRA 적용 및 병합
#model = PeftModel.from_pretrained(base_model, LORA_MODEL)
#model = model.merge_and_unload()        # 최종 가중치 병합
#model.eval()

MODEL_REPO = "kkh27/healthcareLLM_v2"
MODEL_DIR = "./model/healthcareLLM_v2"

# 만약 MODEL_DIR에 모델 파일이 없으면 HuggingFace에서 다운로드
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

# 이후 device_map 적용
model.to("cuda:1")  # 또는 device_map="auto"를 원한다면 위에서 그대로 설정
print("모델 주요 레이어가 올라간 디바이스:", next(model.parameters()).device)
print("모델 로딩 완료!")

# 입력 데이터 형식
class PromptRequest(BaseModel):
    system: str
    question: str
    max_new_tokens: int = 128

# API 엔드포인트
@app.post("/generate")
@torch.inference_mode()
def generate(req: PromptRequest):
    t0 = time.time()
    
    print("== [START] == 프롬프트 수신 및 인퍼런스 시작")

    prompt = (
        "A chat between a curious user and an artificial intelligence assistant.\n"
        "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
        f"System: {req.system}\n"
        f"Human: {req.question}\nAssistant:"
    )
    
    

    print(" 1) 입력 토크나이즈")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    print("모델 디바이스:", next(model.parameters()).device)
    print("입력 텐서 디바이스:", input_ids.device)
    print("input_ids 길이:", input_ids.shape)
    
    print(" 2) 모델 인퍼런스 시작")
    output = model.generate(
        input_ids,
        max_new_tokens=req.max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    print(" 3) 디코딩 및 후처리")
    result = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)

    t1 = time.time()
    print(f"== [DONE] == 인퍼런스 완료! 걸린 시간: {t1-t0:.2f}초\n")
    return {"response": result.strip()}

# 실행 (개발 테스트용)
# 터미널에서: uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
