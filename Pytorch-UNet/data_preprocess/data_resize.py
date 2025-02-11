import os
from PIL import Image

# 데이터 폴더 경로
data_folder = "../CrackFormer-II/datasets/Stone331/val"
image_folder = os.path.join(data_folder, "img")
# mask_folder = os.path.join(data_folder, "masks")

# 리사이즈 크기 (width, height)
target_size = (512, 512)

def process_image(file_path, is_mask=False):
    """
    이미지를 처리하는 함수.
    크기가 360x640이면 90도 회전,
    그 외의 크기는 640x360으로 리사이즈.
    4차원 이미지면 RGB 형식으로 변환.
    """
    with Image.open(file_path) as img:
        original_size = img.size  # (width, height)
        
        if original_size == target_size:
            # 640x360인 경우 그대로 둠
            return
        elif original_size == (512, 512):
            # 360x640인 경우 90도 회전
            img = img.rotate(90, expand=True)
        else:
            # 그 외의 경우 640x360으로 리사이즈
            resample_method = Image.NEAREST if is_mask else Image.BICUBIC
            img = img.resize(target_size, resample=resample_method)
        
        # 변경된 이미지를 덮어쓰기
        img.save(file_path)
        print(f"Processed {file_path}: {original_size} -> {img.size}")

# 이미지와 마스크 폴더에서 모든 파일을 처리
def process_all_images_and_masks():
    # 이미지 파일 처리
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        if os.path.isfile(image_path):
            process_image(image_path)

    # # 마스크 파일 처리
    # for mask_name in os.listdir(mask_folder):
    #     mask_path = os.path.join(mask_folder, mask_name)
    #     if os.path.isfile(mask_path):
    #         process_image(mask_path, is_mask=True)

# 처리 시작
process_all_images_and_masks()
