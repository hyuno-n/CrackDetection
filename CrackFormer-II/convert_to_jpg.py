import os
from PIL import Image
import glob

def convert_to_jpg(dataset_path):
    """
    데이터셋 폴더 내의 모든 이미지 파일을 jpg 형식으로 변환
    
    Args:
        dataset_path: 이미지가 있는 폴더 경로
    """
    # 지원하는 이미지 확장자들 (대소문자 모두 포함)
    valid_extensions = ['.png', '.PNG', '.jpeg', '.JPEG', '.bmp', '.BMP', 
                       '.tiff', '.TIFF', '.gif', '.GIF', '.jpg', '.JPG']
    
# 폴더 내의 모든 파일 검색
    for file_path in glob.glob(os.path.join(dataset_path, '*.*')):
        # 파일명과 확장자 분리
        file_name, ext = os.path.splitext(file_path)
        ext = ext.lower()  # 확장자를 소문자로 변환
        
        # 이미 소문자 .jpg 파일이면서 원본 파일명도 .jpg인 경우 건너뛰기
        if ext == '.jpg' and file_path.endswith('.jpg'):
            continue
            
        # 이미지 파일인 경우에만 처리
        if ext in [e.lower() for e in valid_extensions]:
            try:
                # 이미지 열기
                with Image.open(file_path) as img:
                    # RGB로 변환 (알파 채널이 있는 경우 처리)
                    if img.mode in ('RGBA', 'LA'):
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[-1])
                        img = background
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # jpg로 저장
                    new_path = file_name + '.jpg'
                    img.save(new_path, 'JPEG', quality=95)
                    
                # 원본 파일이 새로운 파일과 다른 경우에만 삭제
                if file_path != new_path:
                    os.remove(file_path)
                    print(f'변환 완료: {file_path} -> {new_path}')
                
            except Exception as e:
                print(f'오류 발생 ({file_path}): {str(e)}')
                continue


if __name__ == '__main__':
    # 데이터셋 경로 설정
    dataset_paths = [
        "./datasets/CrackTree260/test/img/",
        "./datasets/CrackTree260/train/img/",
        "./datasets/CrackTree260/val/img/",
        # 필요한 경우 다른 데이터셋 경로 추가
    ]
    
    for path in dataset_paths:
        if os.path.exists(path):
            print(f"\n{path} 처리 중...")
            convert_to_jpg(path)
        else:
            print(f"\n{path} 경로를 찾을 수 없습니다.") 