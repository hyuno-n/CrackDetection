import os
import shutil
from sklearn.model_selection import train_test_split

# 데이터 경로 설정
base_dir = "Pytorch-UNet/dataset/st"  # 원본 데이터가 있는 폴더
imgs_dir = os.path.join(base_dir, "imgs")
masks_dir = os.path.join(base_dir, "masks")

# 출력 경로 설정
output_dir = "Pytorch-UNet/dataset/Structure"
train_imgs_dir = os.path.join(output_dir, "train/imgs")
train_masks_dir = os.path.join(output_dir, "train/masks")
val_imgs_dir = os.path.join(output_dir, "val/imgs")
val_masks_dir = os.path.join(output_dir, "val/masks")

# 폴더 생성 함수
def create_dirs(dirs):
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

# 출력 폴더 생성
create_dirs([train_imgs_dir, train_masks_dir, val_imgs_dir, val_masks_dir])

# 이미지와 마스크 파일 리스트
img_files = sorted(os.listdir(imgs_dir))
mask_files = sorted(os.listdir(masks_dir))

# 파일 이름 매칭 확인 (이미지와 마스크 이름이 같아야 함)
assert len(img_files) == len(mask_files), "이미지와 마스크의 개수가 다릅니다."
for img, mask in zip(img_files, mask_files):
    assert os.path.splitext(img)[0] == os.path.splitext(mask)[0], f"파일 이름이 맞지 않습니다: {img} vs {mask}"

# Train/Validation 데이터셋 분리 (550:110 비율)
train_imgs, val_imgs, train_masks, val_masks = train_test_split(
    img_files, mask_files, test_size=110/660, random_state=42
)

# 파일 이동 함수
def move_files(file_list, source_dir, target_dir):
    for file_name in file_list:
        shutil.copy(os.path.join(source_dir, file_name), os.path.join(target_dir, file_name))

# Train 데이터 이동
move_files(train_imgs, imgs_dir, train_imgs_dir)
move_files(train_masks, masks_dir, train_masks_dir)

# Validation 데이터 이동
move_files(val_imgs, imgs_dir, val_imgs_dir)
move_files(val_masks, masks_dir, val_masks_dir)

print("데이터셋 분리 완료!")
