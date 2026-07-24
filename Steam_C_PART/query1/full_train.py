import os
import math
import subprocess

def main():
    # 1. macOS 가용 공유 메모리 풀을 90%까지 임시 개방하여 버퍼 안정성 확보
    os.environ["GGML_METAL_SHARED_MEM_PERCENT"] = "90"
    
    model_path = "Qwen/Qwen2.5-7B-Instruct"
    data_dir = "./data"
    adapter_path = "./adapters"
    
    # 2. train.jsonl 행 수를 직접 스캔하여 전수 데이터 크기 파악 (메모리 효율적 방식)
    train_file = os.path.join(data_dir, "train.jsonl")
    if not os.path.exists(train_file):
        print(f"[오류] 학습 데이터 파일이 없습니다: {train_file}")
        return

    print("📂 대용량 전수 JSONL 데이터셋 라인 스캔 중...")
    with open(train_file, "r", encoding="utf-8") as f:
        total_samples = sum(1 for _ in f)
    
    # 3. 하이퍼파라미터 정의 (3 에포크 전수 조율)
    epochs = 3
    batch_size = 1
    learning_rate = "1e-5"
    max_seq_length = "512"
    
    steps_per_epoch = math.ceil(total_samples / batch_size)
    total_steps = steps_per_epoch * epochs
    
    print(f"📊 [데이터 통계] 총 학습 데이터: {total_samples} 개")
    print(f"🔄 [에포크 계획] 1 Epoch = {steps_per_epoch} 스텝 | 총 목표 스텝: {total_steps} 스텝 (총 {epochs}회 순회)")
    print("\n🚀 [전수 멀티 에포크 학습 점화] 하위 모듈 의존성을 격리하여 안전하게 장기 종주를 시작합니다.")
    
    # 4. 검증된 CLI 명령어를 서브프로세스로 바인딩하여 실행
    #cmd = [
    #    "mlx_lm.lora",
    #    "--model", model_path,
    #    "--train",
    #    "--data", data_dir,
    #    "--adapter-path", adapter_path,
    #    "--iters", str(total_steps),
    #    "--batch-size", str(batch_size),
    #    "--learning-rate", learning_rate,
    #    "--max-seq-length", max_seq_length,
    #    "--val-batches", "2",
    #    "--steps-per-report", "100",
    #    "--steps-per-eval", "1000",
    #    "--save-every", "5000"
    #]
    cmd = [
        "mlx_lm.lora",
        "--model", model_path,
        "--train",
        "--data", data_dir,
        "--adapter-path", adapter_path,
        "--iters", str(total_steps),
        "--batch-size", str(batch_size),
        "--learning-rate", learning_rate,
        "--max-seq-length", max_seq_length,
        "--val-batches", "2",
        "--steps-per-report", "100",
        "--steps-per-eval", "1000",
        "--save-every", "5000",
        # 💡 18만 스텝 최종 세이브 파일을 베이스로 지정하여 이어 달리기
        "--resume-adapter-file", "adapters/0180000_adapters.safetensors"
    ]
    
    # 프로세스 실행 및 실시간 로그 스트리밍
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    for line in process.stdout:
        print(line, end="")
        
    process.wait()
    
    if process.returncode == 0:
        print(f"\n🎉 [대성공] 총 {epochs} 에포크 전수 조율 스케줄이 완벽하게 종결되었습니다!")
    else:
        print(f"\n❌ 학습 중 오류가 발생했습니다. 반환 코드: {process.returncode}")

if __name__ == "__main__":
    main()