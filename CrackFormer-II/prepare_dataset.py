import os
from glob import glob
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, default='CrackLS315',
                      choices=['CrackLS315', 'CrackTree260', 'Stone331'])
    return parser.parse_args()

def create_dataset_txt(args):
    """데이터셋 txt 파일 생성"""
    dataset_path = f'./datasets/{args.dataset_type}'
    
    # 각 분할(train/val/test)에 대해 txt 파일 생성
    splits = {'train': 'train', 'valid': 'val', 'test': 'test'}  # txt 파일명과 폴더명 매핑
    
    for txt_name, folder_name in splits.items():
        split_path = os.path.join(dataset_path, folder_name)
        
        # img와 gt 폴더의 파일 목록 가져오기 (.bmp 확장자 지정)
        img_files = sorted(glob(os.path.join(split_path, 'img/*.jpg')))  # 입력 이미지는 jpg
        gt_files = sorted(glob(os.path.join(split_path, 'gt/*.bmp')))   # 라벨은 bmp
        
        if len(img_files) != len(gt_files):
            print(f"Warning: Number of images ({len(img_files)}) and ground truth files ({len(gt_files)}) don't match in {folder_name}")
            continue
        
        # txt 파일 생성
        txt_path = os.path.join(dataset_path, f'{txt_name}.txt')
        with open(txt_path, 'w') as f:
            for img_path, gt_path in zip(img_files, gt_files):
                # 파일명이 매칭되는지 확인 (확장자 제외)
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                gt_name = os.path.splitext(os.path.basename(gt_path))[0]
                
                if img_name == gt_name:
                    # 상대 경로로 변환
                    rel_img_path = os.path.relpath(img_path, dataset_path)
                    rel_gt_path = os.path.relpath(gt_path, dataset_path)
                    f.write(f'{rel_img_path} {rel_gt_path}\n')
                else:
                    print(f"Warning: Mismatch between image {img_name} and ground truth {gt_name}")
        
        print(f'Created {txt_path} with {len(img_files)} pairs')

def check_dataset(args, num_samples=3):
    """데이터셋 샘플 확인"""
    dataset_path = f'./datasets/{args.dataset_type}'
    splits = ['train', 'val', 'test']
    
    for split in splits:
        print(f"\nChecking {split} set...")
        img_dir = os.path.join(dataset_path, split, 'img')
        gt_dir = os.path.join(dataset_path, split, 'gt')
        
        img_files = sorted(glob(os.path.join(img_dir, '*.jpg')))
        gt_files = sorted(glob(os.path.join(gt_dir, '*.bmp')))
        
        if len(img_files) == 0:
            print(f"No images found in {img_dir}")
            continue
            
        # 랜덤하게 샘플 선택
        indices = np.random.choice(len(img_files), min(num_samples, len(img_files)), replace=False)
        
        plt.figure(figsize=(15, 5*len(indices)))
        for idx, sample_idx in enumerate(indices):
            # 이미지 로드
            img_path = img_files[sample_idx]
            gt_path = gt_files[sample_idx]
            
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            
            # 이미지 크기 출력
            print(f"\nSample {idx+1}:")
            print(f"Image path: {img_path}")
            print(f"Image shape: {img.shape}")
            print(f"GT path: {gt_path}")
            print(f"GT shape: {gt.shape}")
            print(f"GT unique values: {np.unique(gt)}")
            
            # 시각화
            plt.subplot(len(indices), 3, idx*3 + 1)
            plt.imshow(img)
            plt.title(f'Input Image\n{os.path.basename(img_path)}')
            plt.axis('off')
            
            plt.subplot(len(indices), 3, idx*3 + 2)
            plt.imshow(gt, cmap='gray')
            plt.title(f'Ground Truth\n{os.path.basename(gt_path)}')
            plt.axis('off')
            
            # Overlay
            overlay = img.copy()
            overlay[gt > 0] = [255, 0, 0]  # 크랙 부분을 빨간색으로 표시
            
            plt.subplot(len(indices), 3, idx*3 + 3)
            plt.imshow(overlay)
            plt.title('Overlay')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(dataset_path, f'{split}_samples.png'))
        plt.close()

def main():
    args = parse_args()
    
    # 데이터셋 기본 경로 확인
    dataset_path = f'./datasets/{args.dataset_type}'
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f'Dataset path not found: {dataset_path}')
    
    # 필요한 폴더 구조 확인
    required_dirs = [
        os.path.join(dataset_path, 'train/img'),
        os.path.join(dataset_path, 'train/gt'),
        os.path.join(dataset_path, 'val/img'),
        os.path.join(dataset_path, 'val/gt'),
        os.path.join(dataset_path, 'test/img'),
        os.path.join(dataset_path, 'test/gt')
    ]
    
    for directory in required_dirs:
        if not os.path.exists(directory):
            raise FileNotFoundError(f'Required directory not found: {directory}')
    
    # txt 파일 생성
    create_dataset_txt(args)
    
    # 데이터셋 확인
    print("\nChecking dataset samples...")
    check_dataset(args)
    
    print(f'\nDataset {args.dataset_type} preparation completed!')
    print(f'Please check the txt files and sample visualizations in {dataset_path}')

if __name__ == '__main__':
    main() 