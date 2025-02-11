import os
import shutil

data_folder = "cam0"
image_folder = os.path.join(data_folder, "images")
mask_folder = os.path.join(data_folder, "labels")

os.makedirs(image_folder, exist_ok=True)
os.makedirs(mask_folder, exist_ok=True)

# data 폴더 안의 파일들을 순회
for filename in os.listdir(data_folder):
    file_path = os.path.join(data_folder, filename)
    
    if os.path.isfile(file_path):
        if filename.lower().endswith(".jpg"):
            # .jpg 파일은 image 폴더로 이동
            shutil.move(file_path, os.path.join(image_folder, filename))
        elif filename.lower().endswith(".txt"):
            # .png 파일은 mask 폴더로 이동
            shutil.move(file_path, os.path.join(mask_folder, filename))

print("파일 정리가 완료되었습니다.")
