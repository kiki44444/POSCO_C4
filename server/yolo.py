import torch
import pandas as pd

def detection(img) :
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    results = model(img)
    image_name = results.pandas().xyxy[0]['name']
    return image_name
