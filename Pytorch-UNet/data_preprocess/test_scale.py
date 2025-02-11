import os
from PIL import Image

# 이미지가 저장된 폴더 경로
image_folder = 'dataset/st/imgs'  # 실제 이미지 폴더 경로로 변경하세요.

# 모델에서 요구하는 채널 수 (예: 3채널 RGB)
expected_channels = 3  # 3 또는 4로 설정 가능

# 폴더 안의 모든 파일을 확인
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

# 채널 수 확인 및 변환하는 함수
def check_and_convert_image_channels(image_path, expected_channels):
    try:
        image = Image.open(image_path)
        original_mode = image.mode  # 원래 모드 확인

        # 현재 채널 수 판단
        if original_mode == 'RGBA':
            channels = 4
        elif original_mode == 'RGB':
            channels = 3
        else:
            channels = len(image.getbands())  # 그 외 모드 고려 (예: L, LA 등)

        # 채널 수 비교 및 변환
        if channels != expected_channels:
            print(f"Converting {image_path}: {channels} channels -> {expected_channels} channels.")
            # RGB로 변환
            image = image.convert('RGB')
            
            # 변환된 이미지를 저장 (덮어쓰기)
            image.save(image_path)
            print(f"Converted {image_path} to {image.mode}.")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# 모든 이미지에 대해 채널 수 확인 및 변환
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    check_and_convert_image_channels(image_path, expected_channels)
