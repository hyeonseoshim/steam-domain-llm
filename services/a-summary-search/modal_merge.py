"""LoRA 병합 1회 실행 — base(Qwen2.5-3B) + 어댑터 → 병합 체크포인트를 데이터 볼륨에 저장.

vLLM 이 병합모델을 서빙하면 토큰당 LoRA 적용(오버헤드)이 사라져 디코드가 base 속도(= 더 빠름).
enable_lora/ max_lora_rank 복잡도도 제거. 1회만: `modal run modal_merge.py`
결과: /data/merged-qwen2.5-3b (steam-part-a-data 볼륨) → gpu 앱이 MERGED_MODEL_DIR 로 로드.
"""

from __future__ import annotations

import pathlib

import modal

REPO = pathlib.Path(__file__).parent
HF_CACHE = "/root/.cache/huggingface"
OUT = "/data/merged-qwen2.5-3b"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers<5", "peft", "accelerate", "safetensors")
    .add_local_dir(str(REPO / "train" / "qwen2.5-3b-lora"), remote_path="/root/adapter",
                   ignore=["checkpoint-*"])
)

data_vol = modal.Volume.from_name("steam-part-a-data", create_if_missing=True)
hf_vol = modal.Volume.from_name("steam-hf-cache", create_if_missing=True)
app = modal.App("steam-part-a-merge")


@app.function(image=image, volumes={"/data": data_vol, HF_CACHE: hf_vol}, timeout=1800)
def merge() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mid = "Qwen/Qwen2.5-3B-Instruct"
    print(f"[merge] base 로드 {mid}")
    base = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16)
    print("[merge] 어댑터 병합(merge_and_unload)")
    merged = PeftModel.from_pretrained(base, "/root/adapter").merge_and_unload()
    print(f"[merge] 저장 → {OUT}")
    merged.save_pretrained(OUT, safe_serialization=True)
    AutoTokenizer.from_pretrained(mid).save_pretrained(OUT)
    data_vol.commit()
    print("[merge] 완료 ✓")


@app.local_entrypoint()
def main() -> None:
    merge.remote()
