import os
import numpy as np
from PIL import Image

# 폴더 경로 설정
train_mask_folder = "dataset/Structure/train/masks/"
val_mask_folder = "dataset/Structure/val/masks/"

# 결과를 저장할 폴더 (같은 위치에 저장)
output_train_mask_folder = "dataset/Structure/train/masks/"
output_val_mask_folder = "dataset/Structure/val/masks/"

# 폴더가 없다면 생성
os.makedirs(output_train_mask_folder, exist_ok=True)
os.makedirs(output_val_mask_folder, exist_ok=True)

# 마스크 이미지 그레이스케일로 처리하고 이진화하는 함수
def process_mask(image_path, output_path):
    # 그레이스케일로 이미지 열기
    mask = Image.open(image_path).convert('L')  # 'L' 모드: 8비트 그레이스케일
    
    # 넘파이 배열로 변환
    mask_array = np.array(mask)

    # 이진화: 0과 255만 존재하도록 설정
    binary_mask = np.where(mask_array > 127, 255, 0).astype(np.uint8)

    # 다시 이미지로 변환
    processed_mask = Image.fromarray(binary_mask, mode='L')

    # 변환된 이미지 저장
    processed_mask.save(output_path, format='PNG')

# train 폴더의 마스크 처리
for filename in os.listdir(train_mask_folder):  
    if filename.endswith(".jpg") or filename.endswith(".png"):  # 이미지 파일만 처리
        image_path = os.path.join(train_mask_folder, filename)
        # 파일 이름에서 확장자 제거 후 '.png'로 변경
        output_path = os.path.join(output_train_mask_folder, os.path.splitext(filename)[0] + '.png')
        process_mask(image_path, output_path)

# val 폴더의 마스크 처리
for filename in os.listdir(val_mask_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # 이미지 파일만 처리
        image_path = os.path.join(val_mask_folder, filename)
        # 파일 이름에서 확장자 제거 후 '.png'로 변경
        output_path = os.path.join(output_val_mask_folder, os.path.splitext(filename)[0] + '.png')
        process_mask(image_path, output_path)

print("마스크 이미지 처리 완료!")
