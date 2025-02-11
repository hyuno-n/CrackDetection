import os
import torch
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

class DataAugmenter:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        # 증강된 데이터 저장 디렉토리 생성
        self.aug_train_img_dir = os.path.join(dataset_path, 'aug_train', 'img')
        self.aug_train_lab_dir = os.path.join(dataset_path, 'aug_train', 'gt')
        
        os.makedirs(self.aug_train_img_dir, exist_ok=True)
        os.makedirs(self.aug_train_lab_dir, exist_ok=True)

    def add_gaussian_noise(self, img, mean=0, std=0.1):
        """이미지에 가우시안 노이즈 추가"""
        np_img = np.array(img).astype(np.float32) / 255.0
        noise = np.random.normal(mean, std, np_img.shape)
        np_img = np.clip(np_img + noise, 0, 1) * 255.0
        return Image.fromarray(np_img.astype(np.uint8))

    def apply_transforms(self, image, gt):
        """이미지와 GT에 동일한 증강을 적용"""
        if random.random() < 0.5:
            image = transforms.functional.hflip(image)
            gt = transforms.functional.hflip(gt)
        if random.random() < 0.5:
            image = transforms.functional.vflip(image)
            gt = transforms.functional.vflip(gt)
        angle = random.uniform(-15, 15)
        image = transforms.functional.rotate(image, angle)
        gt = transforms.functional.rotate(gt, angle)
        
        color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        image = color_jitter(image)
        
        if random.random() < 0.3:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 2.0)))
        if random.random() < 0.3:
            image = self.add_gaussian_noise(image)
        
        return image, gt

    def augment_train_dataset(self, num_augmentations_per_image=3):
        """train 데이터셋의 각 이미지에 대해 증강 수행"""
        train_txt_path = os.path.join(self.dataset_path, 'train.txt')
        with open(train_txt_path, 'r') as f:
            train_pairs = []
            for line in f:
                img_path, gt_path = line.strip().split()
                train_pairs.append((img_path, gt_path))

        print("데이터 증강 시작...")
        for img_rel_path, gt_rel_path in tqdm(train_pairs):
            # 전체 경로 생성
            img_path = os.path.join(self.dataset_path, img_rel_path)
            gt_path = os.path.join(self.dataset_path, gt_rel_path)
            
            try:
                image = Image.open(img_path).convert("RGB")
                gt = Image.open(gt_path).convert("L")
            except FileNotFoundError as e:
                print(f"파일을 찾을 수 없습니다: {e}")
                continue
            
            # 원본 이미지 복사 (상대 경로 유지)
            os.makedirs(os.path.join(self.aug_train_img_dir), exist_ok=True)
            os.makedirs(os.path.join(self.aug_train_lab_dir), exist_ok=True)
            
            img_name = os.path.basename(img_rel_path)
            gt_name = os.path.basename(gt_rel_path)
            
            image.save(os.path.join(self.aug_train_img_dir, img_name))
            gt.save(os.path.join(self.aug_train_lab_dir, gt_name))
            
            # 증강된 이미지 생성
            basename_img = os.path.splitext(img_name)[0]
            basename_gt = os.path.splitext(gt_name)[0]
            ext_img = os.path.splitext(img_name)[1]
            ext_gt = os.path.splitext(gt_name)[1]
            
            for i in range(num_augmentations_per_image):
                aug_image, aug_gt = self.apply_transforms(image, gt)
                aug_image_name = f"{basename_img}_aug_{i}{ext_img}"
                aug_gt_name = f"{basename_gt}_aug_{i}{ext_gt}"
                
                aug_image.save(os.path.join(self.aug_train_img_dir, aug_image_name))
                aug_gt.save(os.path.join(self.aug_train_lab_dir, aug_gt_name))
        
        # 새로운 train.txt 파일 생성
        self.update_train_txt(train_pairs, num_augmentations_per_image)
        print("데이터 증강 완료!")

    def update_train_txt(self, original_pairs, num_augmentations_per_image):
        """증강된 데이터셋에 대한 새로운 train.txt 파일 생성"""
        new_train_txt_path = os.path.join(self.dataset_path, 'augmented_train.txt')
        
        with open(new_train_txt_path, 'w') as f:
            # 원본 이미지 쌍 기록
            for img_path, gt_path in original_pairs:
                f.write(f"{img_path} {gt_path}\n")
                
                # 증강된 이미지 쌍 기록
                img_name = os.path.basename(img_path)
                gt_name = os.path.basename(gt_path)
                basename_img = os.path.splitext(img_name)[0]
                basename_gt = os.path.splitext(gt_name)[0]
                ext_img = os.path.splitext(img_name)[1]
                ext_gt = os.path.splitext(gt_name)[1]
                
                for i in range(num_augmentations_per_image):
                    aug_image_name = f"{basename_img}_aug_{i}{ext_img}"
                    aug_gt_name = f"{basename_gt}_aug_{i}{ext_gt}"
                    f.write(f"aug_train/img/{aug_image_name} aug_train/gt/{aug_gt_name}\n")

if __name__ == "__main__":
    augmenter = DataAugmenter("./datasets/CrackLS315")
    augmenter.augment_train_dataset(num_augmentations_per_image=228)