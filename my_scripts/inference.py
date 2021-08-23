import cv2
import time
from mmdet.apis import init_detector, inference_detector


def main():
    config_file = 'C:/git_projects/mmdetection/configs/yolof/yolof_r50_c5_8x8_1x_coco_my.py'
    checkpoint_file = 'C:/git_projects/pretrained_mmdet/yolof/yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth'
    model = init_detector(config_file, checkpoint_file, device='cpu') # 'cuda:0'
    im = cv2.imread('C:/datasets/my_records/maayan_room_1/frame_000264.png')
    result = inference_detector(model, im)
    model.show_result(im, result, score_thr=0.5, show=True)


if __name__ == '__main__':
    main()

'''
Get data: 
https://drive.google.com/file/d/1RMXKJFyZdN9LlsVPUvmcIx8e5Aiv3fT4/view?usp=sharing

python .\tools\test.py C:/git_projects/mmdetection/configs/yolof/yolof_r50_c5_8x8_1x_coco_my.py C:/git_projects/pretrained_mmdet/yolof/yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth --show-dir C:\datasets\my_records\maayan_room_1\results

python .\tools\test.py C:/git_projects/mmdetection/configs/yolof/yolof_r50_c5_8x8_1x_coco_my.py C:/git_projects/pretrained_mmdet/yolof/yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth --eval recall 

python tools/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--out ${RESULT_FILE}] \
    [--eval ${EVAL_METRICS}] \

python tools/train.py C:/git_projects/mmdetection/configs/yolof/yolof_r50_c5_8x8_1x_coco_my.py
'''