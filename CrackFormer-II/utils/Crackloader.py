import os
import os.path
import torch
from torchvision import transforms
import numpy as np
import scipy.misc as m
import glob
import torch.utils.data as data
import cv2
from torch.utils import data
import PIL.Image as Image
import matplotlib.pyplot as plt


class Crackloader(data.Dataset):

    def __init__(self, txt_path, normalize=False, dataset_type='CrackLS315'):
        """
        Args:
            txt_path (str): 데이터셋 경로가 있는 텍스트 파일 경로
            normalize (bool): 정규화 여부
            dataset_type (str): 'CrackLS315', 'CrackTree260', 'Stone331' 중 하나 선택
        """
        self.dataset_type = dataset_type
        self.imgs = self.read_dataset_paths(txt_path)
        self.normalize = normalize
        
        # 이미지 크기를 512x512로 조정하는 transform 추가
        if self.normalize:
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor()
            ])
            
        # 라벨 이미지도 512x512로 조정
        self.target_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        
        # 데이터셋별 이미지 경로 처리
        img_path = self.get_dataset_path(img_path)
        gt_path = self.get_dataset_path(gt_path)
        # 입력 이미지 로드 및 변환
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        
        # GT(라벨) 이미지 로드 및 변환
        gt = Image.open(gt_path)
        
        # GT 이미지 정보 출력
        gt_np = np.array(gt)
        
        gt = self.target_transform(gt)
        
        # GT를 이진화 (0 또는 1로)
        gt = (gt > 0).float()
        
        return img, gt

    def read_dataset_paths(self, txt_path):
        """텍스트 파일에서 이미지 경로 읽기"""
        imgs = []
        with open(txt_path, 'r') as fh:
            for line in fh:
                line = line.rstrip()
                words = line.split()
                imgs.append((words[0], words[1]))
        return imgs

    def get_dataset_path(self, path):
        """데이터셋 타입에 따른 기본 경로 설정"""
        base_paths = {
            'CrackLS315': './datasets/CrackLS315',
            'CrackTree260': './datasets/CrackTree260',
            'Stone311': './datasets/Stone331'
        }
        
        # 데이터셋 기본 경로 가져오기
        base_path = base_paths.get(self.dataset_type, '')
        
        # 상대 경로를 절대 경로로 변환
        full_path = os.path.join(base_path, path)
        return os.path.normpath(full_path)

    def make_dataset(self, txt_path):
        dataset = []
        index=0
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                # print(index,line)
                index+=1
                line = ''.join(line).strip()
                line_list = line.split(' ')
                dataset.append([line_list[0], line_list[1]])
        return dataset




