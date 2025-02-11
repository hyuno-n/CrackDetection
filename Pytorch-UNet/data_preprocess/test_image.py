import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 이미지 파일 경로
image_path = "dataset/DeepCrack/train/masks/7Q3A9060-1.png"  # 분석할 이미지 경로를 설정하세요.

# 이미지 읽기
image = Image.open(image_path)

# 원본 이미지 출력
plt.imshow(image)
plt.axis("off")
plt.title("Input Image")
plt.show()

# 이미지를 NumPy 배열로 변환
image_array = np.array(image)

# 배열 정보 출력
print("이미지 배열 정보:")
print(f"Shape: {image_array.shape}")  # 배열의 크기 (H, W, C)
print(f"Data type: {image_array.dtype}")  # 데이터 타입 (예: uint8)
print(f"Pixel Values: {image_array}")  # 배열 값 출력 (픽셀 값)

# 배열의 요약 정보 (최대/최소 값 등)
print("\n요약 정보:")
print(f"최소값: {image_array.min()}")
print(f"최대값: {image_array.max()}")

# 그레이스케일 변환 (RGB -> Grayscale)
grayscale_array = np.dot(image_array[...,:3], [0.299, 0.587, 0.114])  # Y = 0.299*R + 0.587*G + 0.114*B

# 그레이스케일 이미지를 0~255 범위로 변환
grayscale_img = Image.fromarray(grayscale_array.astype(np.uint8))

# 그레이스케일 이미지 출력
plt.imshow(grayscale_img, cmap='gray')
plt.axis("off")
plt.title("Grayscale Image")
plt.show()
