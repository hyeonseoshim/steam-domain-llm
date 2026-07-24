import os
import numpy as np
import pandas as pd

emb_folder = os.path.expanduser("~/Downloads/steam_dataset_2025_embeddings")
emb_npy_path = os.path.join(emb_folder, "applications_embeddings.npy")
emb_map_path = os.path.join(emb_folder, "applications_embedding_map.csv")

print("=" * 60)
print("🔍 1. 파일 바이너리 헤더(Hex) 분석")
print("=" * 60)

if os.path.exists(emb_npy_path):
    file_size = os.path.getsize(emb_npy_path)
    print(f"[*] 파일 크기: {file_size / (1024**2):.2f} MB ({file_size} 바이트)")
    
    # 첫 64바이트 추출하여 포맷 확인
    with open(emb_npy_path, 'rb') as f:
        header = f.read(64)
        print(f"[*] 헤더 (Hex): {header.hex()}")
        print(f"[*] 헤더 (Text): {header}\n")
else:
    print("[오류] 파일을 찾을 수 없습니다.")

print("=" * 60)
print("🛠️ 2. Raw Float32 바이너리 스트리밍 역산 테스트")
print("=" * 60)

try:
    df_map = pd.read_csv(emb_map_path)
    total_records = len(df_map)
    print(f"[*] 매핑 CSV 기준 총 게임 수: {total_records} 개")
    
    # BGE-M3 1024차원 float32 (차원당 4바이트) 가정 계산
    expected_vector_size = 1024
    
    print("[*] NumPy 메모리 맵(fromfile)으로 재시도 중...")
    # 헤더 오류가 나는 경우 NumPy 구조 정보가 깨진 것이므로 
    # 데이터를 처음부터 float32 가공 데이터 스트림으로 직접 읽어옵니다.
    data = np.memmap(emb_npy_path, dtype='float32', mode='r')
    
    # 실제 파일 크기와 계산된 차원이 맞는지 검증
    actual_dimensions = len(data) // total_records
    print(f"[*] 계산된 실제 벡터 차원수: {actual_dimensions} 차원")
    
    # 1024차원이 맞다면 구조를 재배열합니다.
    if actual_dimensions == expected_vector_size:
        emb_matrix = data.reshape(total_records, expected_vector_size)
        print("✅ [성공] 임베딩 행렬을 성공적으로 복구 및 로드했습니다!")
        print(f"[*] 최종 Matrix Shape: {emb_matrix.shape}")
        print(f"[*] 첫 번째 게임의 벡터 상위 5개 값: {emb_matrix[0][:5]}")
    else:
        # 혹시 앞에 npy 헤더 잔재(예: 80바이트 또는 128바이트)가 밀려있을 경우의 예외처리
        print("[안내] 헤더 오프셋이 존재할 수 있습니다. 오프셋 역산 시도...")
        offset = file_size - (total_records * expected_vector_size * 4)
        print(f"[*] 추정되는 헤더 오프셋 크기: {offset} 바이트")
        
        if offset > 0:
            emb_matrix = np.memmap(emb_npy_path, dtype='float32', mode='r', offset=offset)
            emb_matrix = emb_matrix.reshape(total_records, expected_vector_size)
            print("✅ [성공] 오프셋 보정 후 임베딩 행렬 복구 완료!")
            print(f"[*] 최종 Matrix Shape: {emb_matrix.shape}")
        else:
            print("❌ 차원이 맞지 않습니다. 헤더의 실제 Hex 값을 확인해야 합니다.")

except Exception as e:
    print(f"❌ 복구 실패 원인: {e}")