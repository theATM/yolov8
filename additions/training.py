from ultralytics import YOLO
import os
print(os.getcwd())

# Load a model
model_path = "./pretrained/yolov8s.pt" #"./runs/detect/train14/weights/best.pt" # "yolov8n.pt"
model = YOLO(model_path) # load a pretrained model (recommended for training)
# Use the model
model.train(data="./rds-robo.yaml", epochs=10, batch=8, freeze=[10])  # train the model
model.val()  # evaluate model performance on the validation set
#model("./datasets/RSD-GOD-TXT/images/test",save=True)  # predict on an image

# CLI:
# yolo detect predict model=yolov8n.pt source="https://ultralytics.com/images/bus.jpg" save=true
# yolo predict model="./runs/detect/train14/weights/best.pt" source="./airbase_8749.jpg" save=true
#  yolo predict model="./runs/detect/train14/weights/best.pt" source="datasets/RSD-GOD-TXT/images/test"