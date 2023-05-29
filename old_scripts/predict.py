#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ultralytics import YOLO
#  import argparse

#  parser = argparse.ArgumentParser()
#  parser.add_argument('input_img')
#  parser.add_argument('output_dir')
#  args = parser.parse_args()

model = YOLO('/home/shaodi/Work/Embryo/yolov8/yolov8m/train/weights/best.pt')
#  model.predict(args.input_img, save=True, imgsz=512, conf=0.5, device='0,1,2')

#  parser.add_argument('input')
#  parser.add_argument('output')
#  args = parser.parse_args()
#  result = model(args.input)
#  result.save(args.output)


results = model.predict('/home/shaodi/Work/Embryo/yolov8/data/aligned_crop/*.png' , \
              save=True, \
              conf=0.4 ,\
              save_txt=True, \
              save_conf=True, \
              save_crop=True, \
              line_thickness=1, \
              imgsz=512, \
              show_labels=False, \
              max_det=1000, \
              device='0,1,2', \
              )
