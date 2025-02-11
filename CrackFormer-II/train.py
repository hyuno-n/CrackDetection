import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from nets.crackformerII import crackformer
from utils.Crackloader import Crackloader
from utils.lossFunctions import cross_entropy_loss_RCF  # 손실 함수 임포트
from utils.Validator import Validator  # Validator 임포트
import argparse
import os
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dataset_type', type=str, default='CrackLS315',
                      choices=['CrackLS315', 'CrackTree260', 'Stone331'])
    parser.add_argument('--train_txt', type=str, 
                      default='./datasets/CrackLS315/train.txt')
    parser.add_argument('--val_txt', type=str, 
                      default='./datasets/CrackLS315/valid.txt')
    parser.add_argument('--valid_img_dir', type=str, 
                      default='./datasets/CrackLS315/val/img')
    parser.add_argument('--valid_lab_dir', type=str, 
                      default='./datasets/CrackLS315/val/gt')
    parser.add_argument('--valid_result_dir', type=str, default='./results/valid/')
    parser.add_argument('--valid_log_dir', type=str, default='./logs/valid/')
    parser.add_argument('--best_model_dir', type=str, default='./checkpoints/best/')
    parser.add_argument('--save_path', type=str, default='./checkpoints')
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()

def train_one_epoch(model, train_loader, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0
    
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}') as pbar:
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            
            # forward pass
            outs = model(images)
            side5, side4, side3, side2, side1, fused = outs
            
            # RCF loss 계산
            loss5 = cross_entropy_loss_RCF(side5, targets)
            loss4 = cross_entropy_loss_RCF(side4, targets)
            loss3 = cross_entropy_loss_RCF(side3, targets)
            loss2 = cross_entropy_loss_RCF(side2, targets)
            loss1 = cross_entropy_loss_RCF(side1, targets)
            loss_fused = cross_entropy_loss_RCF(fused, targets)
            
            # 각 side output과 fusion output의 가중치 설정
            S_side = [0.5, 0.5, 0.5, 1.2, 1.2]  # 마지막 두 개의 side output 가중치 증가
            S_fuse = 1.5  # fusion output의 가중치를 높게 설정

            # 총 loss 계산
            loss = (
                S_side[0] * loss1 +
                S_side[1] * loss2 +
                S_side[2] * loss3 +
                S_side[3] * loss4 +
                S_side[4] * loss5 +
                S_fuse * loss_fused
            )
            loss.backward()
            optimizer.step()
            
            # 결과 저장 (100 배치마다)
            if batch_idx % 100 == 0:
                save_dir = os.path.join('./results/train', str(epoch))
                os.makedirs(save_dir, exist_ok=True)
                
                # Validator 스타일로 출력 처리
                output = torch.sigmoid(fused)  # sigmoid 적용
                out_clone = output.clone()
                img_fused = np.squeeze(out_clone.cpu().detach().numpy(), axis=0)
                img_fused = np.transpose(img_fused, (1, 2, 0))
                cv2.imwrite(os.path.join(save_dir, f'batch_{batch_idx}.jpg'), img_fused * 255.0)
            
            # 메모리 정리
            del side5, side4, side3, side2, side1, fused
            del loss5, loss4, loss3, loss2, loss1, loss_fused
            torch.cuda.empty_cache()
            
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}'
            })
            pbar.update(1)
    
    return total_loss / len(train_loader)

def check_dataloader(train_loader, save_dir='./dataloader_check'):
    """데이터 로더에서 불러온 배치 확인"""
    os.makedirs(save_dir, exist_ok=True)
    
    print("\nChecking DataLoader...")
    for batch_idx, (images, targets) in enumerate(train_loader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"Batch size: {images.size(0)}")
        print(f"Image shape: {images.shape}")
        print(f"Target shape: {targets.shape}")
        print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"Target unique values: {torch.unique(targets).tolist()}")
        print(f"Number of positive pixels in GT: {(targets > 0.5).sum().item()}")
        print(f"Percentage of positive pixels: {(targets > 0.5).sum().item() / targets.numel() * 100:.2f}%")
        
        # 시각화 및 저장
        n_samples = min(4, images.size(0))
        plt.figure(figsize=(15, 4*n_samples))
        
        for idx in range(n_samples):
            # 원본 이미지
            img = images[idx].permute(1, 2, 0).numpy()
            img = (img * 0.5 + 0.5)
            img = np.clip(img, 0, 1)
            
            # GT 마스크
            mask = targets[idx].squeeze().numpy()
            
            # Overlay
            overlay = img.copy()
            overlay[mask > 0.5] = [1, 0, 0]
            
            plt.subplot(n_samples, 3, idx*3 + 1)
            plt.imshow(img)
            plt.title(f'Image {idx+1}\n{img.shape}')
            plt.axis('off')
            
            plt.subplot(n_samples, 3, idx*3 + 2)
            plt.imshow(mask, cmap='gray')
            plt.title(f'GT {idx+1}\nUnique: {np.unique(mask)}')
            plt.axis('off')
            
            plt.subplot(n_samples, 3, idx*3 + 3)
            plt.imshow(overlay)
            plt.title(f'Overlay {idx+1}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'batch_{batch_idx+1}_samples.png'))
        plt.close()
        
        if batch_idx == 2:  # 처음 3개 배치만 확인
            break
    
    print(f"\nSaved dataloader check results to {save_dir}")

def print_model_summary(model, input_size=(1, 3, 512, 512)):
    """모델 구조와 파라미터 정보를 출력"""
    def sizeof_fmt(num, suffix='B'):
        for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
            if abs(num) < 1024.0:
                return f"{num:3.1f}{unit}{suffix}"
            num /= 1024.0
        return f"{num:.1f}Yi{suffix}"
    
    print("\n" + "="*50)
    print("Model Summary:")
    print("="*50)
    
    # 총 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 모델 크기 계산 (대략적)
    param_size = sizeof_fmt(total_params * 4)  # 4 bytes per parameter
    
    print(f"Architecture: {model.__class__.__name__}")
    print(f"Input size: {input_size}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Estimated model size: {param_size}")
    
    # 레이어별 파라미터 수
    print("\nLayer-wise parameters:")
    print("-"*50)
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        print(f"{name}: {num_params:,} parameters")
    
    print("="*50 + "\n")

def main():
    args = parse_args()
    
    # CUDA 메모리 설정 추가
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    
    # GPU 메모리 할당 설정
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # GradScaler 초기화
    scaler = torch.cuda.amp.GradScaler()
    
    model = crackformer().cuda()
    
    # 모델 서머리 출력
    print_model_summary(model)
    
    # 학습 설정 출력
    print("Training Configuration:")
    print("-"*50)
    print(f"Dataset: {args.dataset_type}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Total epochs: {args.epochs}")
    print(f"Device: {device}")
    print(f"Training data: {args.train_txt}")
    print(f"Validation data: {args.val_txt}")
    print("-"*50 + "\n")
    
    train_dataset = Crackloader(
        txt_path=args.train_txt,
        normalize=True,
        dataset_type=args.dataset_type
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=0,
        pin_memory=True
    )
    
    # 데이터 로더 체크 추가
    check_dataloader(train_loader)
    
    # 계속 진행할지 사용자 확인
    response = input("\nDataloader check completed. Continue with training? (y/n): ")
    if response.lower() != 'y':
        print("Training aborted.")
        return
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50, 100], gamma=0.1)
    
    writer = SummaryWriter(os.path.join(args.save_path, 'logs'))
    
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.valid_log_dir, exist_ok=True)
    os.makedirs(args.best_model_dir, exist_ok=True)
    
    # Validator 객체를 한 번만 생성
    valid_img_dir = os.path.join(f'./datasets/{args.dataset_type}/val/img')
    valid_lab_dir = os.path.join(f'./datasets/{args.dataset_type}/val/gt')
    
    validator = Validator(
        valid_img_dir=valid_img_dir,
        valid_lab_dir=valid_lab_dir,
        valid_result_dir=args.valid_result_dir,
        valid_log_dir=args.valid_log_dir,
        best_model_dir=args.best_model_dir,
        net=model,
        normalize=True
    )
    
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch)
        # validate 함수 수정
        os.makedirs(os.path.join(args.valid_result_dir, str(epoch)), exist_ok=True)
        val_loss = validator.validate(epoch)
        
        scheduler.step(val_loss)
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
            }, os.path.join(args.save_path, 'best_model.pth'))
        
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, os.path.join(args.save_path, f'checkpoint_epoch_{epoch+1}.pth'))
        
        print(f'Epoch {epoch+1}/{args.epochs}')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print('-' * 50)
    
    writer.close()

if __name__ == '__main__':
    main()