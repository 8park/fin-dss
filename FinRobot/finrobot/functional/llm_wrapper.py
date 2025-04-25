# FinRobot/finrobot/functional/llm_wrapper.py

import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ★ 사용할 HF 모델 이름
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

# 1) 모델과 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
)
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    max_new_tokens=512,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
    return_full_text=False
)

def generate_strategy_llama(role: str, market_state: dict) -> dict:
    """
    로컬 Llama로 전략 JSON을 생성합니다.
    출력 예시: {"risk_pref":0.3, "target_return":0.1}
    """
    prompt = (
        f"You are a financial strategist.\n"
        f"Role: {role}\n"
        f"Market state: {json.dumps(market_state)}\n"
        f"Please provide a JSON with keys \"risk_pref\" (0–1) and \"target_return\" (0–1)."
    )
    response = llm_pipeline(prompt)[0]["generated_text"]
    raw = response.strip()
    # Extract only the JSON substring
    json_str = raw[raw.find("{"): raw.rfind("}")+1]
    return json.loads(json_str)
