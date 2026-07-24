import os
import pandas as pd
import json

# 맥북의 '다운로드' 폴더 내 실제 저장된 폴더명 매핑
# (사용자 계정명을 자동으로 추적하므로 별도 수정 없이 실행 가능합니다.)
folder_1_path = os.path.expanduser("~/Downloads/steam_dataset_2025_csv")
folder_2_path = os.path.expanduser("~/Downloads/steam_dataset_2025_embeddings")

def analyze_folder(folder_path, folder_name):
    print("=" * 60)
    print(f"📂 [{folder_name}] 폴더 분석 결과 (경로: {folder_path})")
    print("=" * 60)
    
    if not os.path.exists(folder_path):
        print("[오류] 폴더를 찾을 수 없습니다. 경로를 확인해주세요.\n")
        return

    # 1. 파일 목록 출력
    files = [f for f in os.listdir(folder_path) if not f.startswith('.')]
    print(f"[*] 포함된 파일 목록 ({len(files)}개):")
    for file in files[:15]:  # 너무 많을 수 있으니 상위 15개만 출력
        file_size = os.path.getsize(os.path.join(folder_path, file)) / (1024**2) # MB 단위
        print(f"  - {file} ({file_size:.2f} MB)")
    if len(files) > 15:
        print(f"  ... 외 {len(files) - 15}개 파일 더 있음")
    
    # 2. 대표 파일 샘플 분석 (CSV 또는 JSON)
    print("\n[*] 대표 파일 내부 구조 샘플 파악:")
    csv_files = [f for f in files if f.endswith('.csv')]
    json_files = [f for f in files if f.endswith('.json')]
    
    # CSV 샘플 읽기
    if csv_files:
        target_csv = os.path.join(folder_path, csv_files[0])
        print(f"  [CSV 샘플] 파일명: {csv_files[0]}")
        try:
            df = pd.read_csv(target_csv, nrows=2)
            print(f"  - 컬럼 목록: {list(df.columns)}")
            print("  - 데이터 샘플:")
            print(df.to_string())
        except Exception as e:
            print(f"  - CSV 로드 중 오류 발생: {e}")
            
    # JSON 샘플 읽기
    if json_files:
        target_json = os.path.join(folder_path, json_files[0])
        print(f"\n  [JSON 샘플] 파일명: {json_files[0]}")
        try:
            with open(target_json, 'r', encoding='utf-8') as f:
                # 첫 몇 줄만 읽거나 가볍게 파싱
                head = [next(f) for _ in range(5)]
                print("  - 파일 시작 부분 (텍스트):")
                print("".join(head))
        except Exception as e:
            print(f"  - JSON 로드 중 오류 발생: {e}")
    print("\n")

# 실행
analyze_folder(folder_1_path, "폴더 1")
analyze_folder(folder_2_path, "폴더 2")