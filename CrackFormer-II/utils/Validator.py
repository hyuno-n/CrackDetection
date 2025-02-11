import cv2
import os.path
import torch
from torchvision import transforms
import numpy as np
import glob
from torch.autograd import Variable
import datetime
from utils.lossFunctions import cross_entropy_loss_RCF 
class Validator(object):

    def __init__(self, valid_img_dir, valid_lab_dir, valid_result_dir, valid_log_dir, best_model_dir,
                 net,image_format = "jpg",lable_format = "bmp", normalize = False):

        self.valid_img_dir = valid_img_dir  # 验证集的路径
        self.valid_lab_dir = valid_lab_dir  # 验证集GT的路径
        self.valid_res_dir = valid_result_dir # 验证集生成结果的路径
        self.best_model_dir = best_model_dir
        self.valid_log_dir = valid_log_dir + "/valid.txt" # 验证集测试指标的路径
        self.image_format = image_format
        self.lable_format = lable_format
        self.ods = 0
        self.ois = 0
        if os.path.exists(self.best_model_dir)==False:
            os.makedirs(self.best_model_dir)
        self.net = net
        self.normalize = normalize
        # 数值归一化到[-1, 1]
        if self.normalize:
            self.transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            self.transforms = transforms.ToTensor()


    def make_dir(self):
        try:
            if not os.path.exists(self.valid_res_dir):
                os.makedirs(self.valid_res_dir)
        except:
            print("创建valid_res文件失败")


    def make_dataset(self, epoch_num):
        pred_imgs, gt_imgs = [], []
        for pred_path in glob.glob(os.path.join(self.valid_res_dir + str(epoch_num) + "/", "*." + self.image_format)):

            gt_path = os.path.join(self.valid_lab_dir, os.path.basename(pred_path)[:-4] + "." + self.lable_format)

            gt_img = self.imread(gt_path, thresh=80)
            pred_img = self.imread(pred_path, gt_img)
            gt_imgs.append(gt_img)
            pred_imgs.append(pred_img)

        return pred_imgs, gt_imgs

    def imread(self, path, rgb2gray=None, load_size=0, load_mode=cv2.IMREAD_GRAYSCALE, convert_rgb=False, thresh=-1):
        im = cv2.imread(path, load_mode)
        if convert_rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if load_size > 0:
            im = cv2.resize(im, (load_size, load_size), interpolation=cv2.INTER_CUBIC)
        if thresh > 0:
            _, im = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY)
        else:
            im = ((rgb2gray == 0) + (rgb2gray == 255)) * im
        return im

    def get_statistics(self, pred, gt):
        """
        return tp, fp, fn
        """
        tp = np.sum((pred == 1) & (gt == 1))
        fp = np.sum((pred == 1) & (gt == 0))
        fn = np.sum((pred == 0) & (gt == 1))
        return [tp, fp, fn]

    # 计算 ODS 方法
    def cal_prf_metrics(self, pred_list, gt_list, thresh_step=0.01):
        final_accuracy_all = []
        for thresh in np.arange(0.0, 1.0, thresh_step):
            statistics = []

            for pred, gt in zip(pred_list, gt_list):
                gt_img = (gt / 255).astype('uint8')
                pred_img = ((pred / 255) > thresh).astype('uint8')
                # calculate each image
                statistics.append(self.get_statistics(pred_img, gt_img))

            # get tp, fp, fn
            tp = np.sum([v[0] for v in statistics])
            fp = np.sum([v[1] for v in statistics])
            fn = np.sum([v[2] for v in statistics])

            # calculate precision
            p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
            # calculate recall
            r_acc = tp / (tp + fn)
            # calculate f-score
            final_accuracy_all.append([thresh, p_acc, r_acc, 2 * p_acc * r_acc / (p_acc + r_acc)])

        return final_accuracy_all

    # 计算 OIS 方法
    def cal_ois_metrics(self, pred_list, gt_list, thresh_step=0.01):
        final_acc_all = []
        eps = 1e-7 
        
        for pred, gt in zip(pred_list, gt_list):
            statistics = []
            for thresh in np.arange(0.0, 1.0, thresh_step):
                gt_img = (gt / 255).astype('uint8')
                pred_img = (pred / 255 > thresh).astype('uint8')
                tp, fp, fn = self.get_statistics(pred_img, gt_img)
                p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp + eps)
                r_acc = tp / (tp + fn + eps)

                if p_acc + r_acc == 0:
                    f1 = 0
                else:
                    f1 = 2 * p_acc * r_acc / (p_acc + r_acc + eps)
                statistics.append([thresh, f1])
            max_f = np.amax(statistics, axis=0)
            final_acc_all.append(max_f[1])
        return np.mean(final_acc_all)

    # 计算 AP 方法
    def cal_ap_metrics(self, pred_list, gt_list, thresh_step=0.01):
        eps = 1e-7
        precisions = []
        recalls = []
        
        # 각 threshold에서의 precision과 recall 계산
        for thresh in np.arange(0.0, 1.0, thresh_step):
            statistics = []
            
            # 각 이미지에 대해 통계 수집
            for pred, gt in zip(pred_list, gt_list):
                gt_img = (gt / 255).astype('uint8')
                pred_img = ((pred / 255) > thresh).astype('uint8')
                # calculate each image
                statistics.append(self.get_statistics(pred_img, gt_img))
            
            # get tp, fp, fn
            tp = np.sum([v[0] for v in statistics])
            fp = np.sum([v[1] for v in statistics])
            fn = np.sum([v[2] for v in statistics])
            
            # calculate precision
            precision = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp + eps)
            # calculate recall
            recall = tp / (tp + fn + eps)
            
            precisions.append(precision)
            recalls.append(recall)
        
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        
        # recall 기준으로 정렬
        sort_idx = np.argsort(recalls)
        recalls = recalls[sort_idx]
        precisions = precisions[sort_idx]
        
        # precision 보정 (monotonically decreasing)
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])
        
        # AP 계산 (면적)
        ap = 0.0
        for i in range(1, len(recalls)):
            ap += (recalls[i] - recalls[i-1]) * precisions[i]
            
        return ap

    def validate(self, epoch_num):
        print('검증 시작')
        image_list = os.listdir(self.valid_img_dir)
        total_val_loss = 0.0
        batch_count = 0

        # 결과 저장 디렉토리 생성
        if not os.path.exists(self.valid_res_dir + str(epoch_num)):
            os.makedirs(self.valid_res_dir + str(epoch_num))
        
        self.net.eval()
        with torch.no_grad():
            for image_name in image_list:
                # 이미지 로드
                image = os.path.join(self.valid_img_dir, image_name)
                image = cv2.imread(image)
                x = Variable(self.transforms(image))
                x = x.unsqueeze(0)
                x = x.cuda()
                
                # Ground Truth 로드
                gt_path = os.path.join(self.valid_lab_dir, os.path.splitext(image_name)[0] + "." + self.lable_format)
                gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                gt_tensor = torch.from_numpy(gt_image).float() / 255.0
                gt_tensor = gt_tensor.unsqueeze(0).unsqueeze(0).cuda()
                
                # 모델 추론
                outs = self.net.forward(x)               
                side5, side4, side3, side2, side1, fused = outs
            
                # RCF loss 계산
                loss5 = cross_entropy_loss_RCF(side5, gt_tensor)
                loss4 = cross_entropy_loss_RCF(side4, gt_tensor)
                loss3 = cross_entropy_loss_RCF(side3, gt_tensor)
                loss2 = cross_entropy_loss_RCF(side2, gt_tensor)
                loss1 = cross_entropy_loss_RCF(side1, gt_tensor)
                loss_fused = cross_entropy_loss_RCF(fused, gt_tensor)
                
                # 각 side output과 fusion output의 가중치 설정
                S_side = [0.5, 0.5, 0.5, 1.2, 1.2]  # 마지막 두 개의 side output 가중치 증가
                S_fuse = 1.5  # fusion output의 가중치를 높게 설정

                # 총 loss 계산
                val_loss = (
                    S_side[0] * loss1 +
                    S_side[1] * loss2 +
                    S_side[2] * loss3 +
                    S_side[3] * loss4 +
                    S_side[4] * loss5 +
                    S_fuse * loss_fused
                )
                total_val_loss += val_loss.item()
                batch_count += 1
                
                # 결과 이미지 저장
                output = torch.sigmoid(fused)
                out_clone = output.clone()
                img_fused = np.squeeze(out_clone.cpu().detach().numpy(), axis=0)
                img_fused = np.transpose(img_fused, (1, 2, 0))
                cv2.imwrite(self.valid_res_dir + str(epoch_num) + '/' + image_name, img_fused * 255.0)

            # 평균 validation loss 계산
            avg_val_loss = total_val_loss / batch_count if batch_count > 0 else 0

        # 기존 메트릭 계산
        img_list, gt_list = self.make_dataset(epoch_num)
        final_results = self.cal_prf_metrics(img_list, gt_list, 0.01)
        final_ois = self.cal_ois_metrics(img_list, gt_list, 0.01)
        ap_score = self.cal_ap_metrics(img_list, gt_list, 0.01)
        max_f = np.amax(final_results, axis=0)
        
        if max_f[3] > self.ods:
            self.ods = max_f[3]
            self.ois = final_ois
            ods_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-") + str(max_f[3])[0:5]
            print('save ' + ods_str)
            torch.save(self.net.state_dict(), self.best_model_dir + ods_str + ".pth")
        
        with open(self.valid_log_dir, 'a', encoding='utf-8') as fout:
            line = "epoch:{} | Val Loss:{:.6f} | ODS:{:.6f} | OIS:{:.6f} | AP:{:.6f} | max ODS:{:.6f} | max OIS:{:.6f} " \
                .format(epoch_num, avg_val_loss, max_f[3], final_ois, ap_score, self.ods, self.ois) + '\n'
            fout.write(line)
        print("epoch={} Val Loss:{:.6f} | ODS:{:.6f} | OIS:{:.6f} | AP:{:.6f} | max ODS:{:.6f} | max OIS:{:.6f}"
              .format(epoch_num, avg_val_loss, max_f[3], final_ois, ap_score, self.ods, self.ois))
        
        return avg_val_loss