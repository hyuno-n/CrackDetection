import cv2
import numpy as np
import matplotlib.pyplot as plt
from albumentations import ElasticTransform, Compose

# ElasticTransform 설정
transform = Compose(
    [ElasticTransform(alpha=60, sigma=5, p=1.0)],
    additional_targets={'mask': 'mask'}  # 마스크와 동일한 변환 적용
)

# 이미지와 마스크 불러오기
image_path = 'Pytorch-UNet/images/dc_test.jpg'  # 이미지 경로
mask_path = 'Pytorch-UNet/images/dc_test_mask.png'    # 마스크 경로

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 마스크는 흑백으로 로드

# ElasticTransform 적용
augmented = transform(image=image, mask=mask)
transformed_image = augmented['image']
transformed_mask = augmented['mask']

# 결과 시각화
plt.figure(figsize=(12, 6))

# 원본 이미지
plt.subplot(2, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

# 변형된 이미지
plt.subplot(2, 2, 2)
plt.imshow(transformed_image)
plt.title("Transformed Image")
plt.axis("off")

# 원본 마스크
plt.subplot(2, 2, 3)
plt.imshow(mask, cmap='gray')
plt.title("Original Mask")
plt.axis("off")

# 변형된 마스크
plt.subplot(2, 2, 4)
plt.imshow(transformed_mask, cmap='gray')
plt.title("Transformed Mask")
plt.axis("off")

plt.tight_layout()
plt.show()
