from ultralytics import YOLO
import os
model = YOLO('yolov8x.pt')
mouse_imgs = os.path.join(os.getcwd() , 'mouse')
output_dir = os.path.join(os.getcwd() , 'output')

results = model.predict(source = mouse_imgs ,conf=0.25 , save_dir = output_dir , save = True )
