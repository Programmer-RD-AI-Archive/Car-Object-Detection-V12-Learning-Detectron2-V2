import random
import cv2
import json
from detectron2.data.catalog import Metadata
from detectron2.engine import DefaultPredictor, DefaultTrainer
import os
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import ColorMode
import matplotlib.pyplot as plt
from tqdm import tqdm
from detectron2.structures import BoxMode
from detectron2.config import get_cfg
from detectron2 import model_zoo
import wandb
import pandas as pd
import numpy as np
import torch
import torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
PROJECT_NAME = 'Car-Object-Detection-V12-Learning-Detectron2-V2'

data = pd.read_csv('data.csv')
info = data.iloc[0]
img = cv2.imread('./data/' + info['image'])
xmin, xmax, ymin, ymax = info['xmin'], info['xmax'], info['ymin'], info['ymax']
x = xmin
y = ymin
w = xmax - xmin
h = ymax - ymin
# crop = img[y:y+h,x:x+w]
# plt.figure(figsize=(12, 6))
# plt.imshow(crop)
# plt.savefig('0.png')
# plt.close()
# cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
# plt.figure(figsize=(12, 6))
# plt.imshow(img)
# plt.savefig('1.png')
# plt.close()


def load_data():
    new_data = []
    for i in len(data):
        info = data.iloc[i]
        height, width = cv2.imread(info['image']).shape[:2]
        file_name = "./data/" + info['image']
        record = {}
        record['file_name'] = file_name
        record['height'] = height
        record['width'] = width
        record['image_id'] = i
        record['annotations'] = {
            'bbox': [info['xmin'], info['ymin'], info['xmax'], info['ymax']], 'bbox_mode': BoxMode.XYXY_ABS,
            'category_id': 0, 'class_id': 0
        }
        new_data.append(record)
    return new_data


labels = ['car']
DatasetCatalog.register('data', lambda: load_data())
metadata = MetadataCatalog.get('data').set(thing_classes=labels)
model = "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(model))
cfg.DATASETS.TRAIN = ('data')
cfg.DATASETS.TEST = ()
cfg.SOLVER.STEPS = []
cfg.SOLVER.MAX_ITER = 1250
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.BASE_LR = 0.00025
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(labels)
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
predictor = DefaultPredictor(cfg)
evaluator = COCOEvaluator('data', output_dir='./output')
data_loader = build_detection_test_loader(cfg, 'data')
metrics = inference_on_dataset(trainer.model, data_loader, evaluator)
print(metrics)
img = cv2.imread('./data/' + info['image'])
pred = predictor(img)
v = Visualizer(img[:, :, ::-1])
v.draw_instance_predictions(pred)
img = v.get_image()[:, :, ::-1]
plt.figure(figsize=(12, 6))
plt.imshow(img)
plt.savefig('./2.png')
torch.save(cfg, 'cfg.pt')
torch.save(cfg, 'cfg.pth')
torch.save(predictor, 'predictor.pt')
torch.save(predictor, 'predictor.pth')
torch.save(evaluator, 'evaluator.pt')
torch.save(evaluator, 'evaluator.pth')
torch.save(v, 'img.pt')
torch.save(v, 'img.pth')
torch.save(model, 'model.pt')
torch.save(model, 'model.pth')
torch.save(labels, 'labels.pt')
torch.save(labels, 'labels.pth')
torch.save(metrics, 'metrics.pt')
torch.save(metrics, 'metrics.pth')
