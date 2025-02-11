from utils.utils import *
from utils.Validator import *
from utils.Crackloader import *
from nets.crackformerII import crackformer
import os

def Test():
    netName = "crackformer"
    valid_log_dir = "./log/" + netName
    datasetName = "CrackLS315"
    # datasetName="Stone331"
    # datasetName='CrackTree260'
    valid_img_dir = "./datasets/" + datasetName + "/test/img/"
    valid_lab_dir = "./datasets/" + datasetName + "/test/gt/"
    if os.path.exists(valid_img_dir)==False:
        os.makedirs(valid_img_dir)
    if os.path.exists(valid_lab_dir)==False:
        os.makedirs(valid_lab_dir)
    # pretrain_dir="model/crack260.pth"
    pretrain_dir="model/crackformer/crack315.pth"
    # pretrain_dir='model/crack537.pth'
    valid_result_dir = "./datasets/" + datasetName + "/test/result/"
    valid_log_dir = "./logs/valid/"
    best_model_dir = "./checkpoints/best/"
    
    # 모델 초기화
    net = crackformer().cuda() 
    
    # 모델 로드
    model_path = pretrain_dir
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    else:
        print(f"No model found at {model_path}")
        return
    
    # Validator 초기화 
    validator = Validator(
        valid_img_dir=valid_img_dir,
        valid_lab_dir=valid_lab_dir,
        valid_result_dir=valid_result_dir,
        valid_log_dir=valid_log_dir,
        best_model_dir=best_model_dir,
        net=net,
        image_format="jpg",
        lable_format="bmp",
        normalize=True
    )
    
    # 검증 실행
    validator.validate('0')

if __name__ == '__main__':
    Test()