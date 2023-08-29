from ultralytics import YOLO
import os

# Load a model
model_path = "./runs/best/Robo200/weights/best.pt" # load a checkpoint
model = YOLO(model_path)
# evaluate model performance on the validation set - change the validation set path to change the validated set:
results = model.val(val=True,data="./rds-robo.yaml",coco_annot='./datasets/RDS-ROBO/valid/instances_val.json',save_json=True,plots=True, box_fusion = 'cp_clustering')
# predict on on test images
#model("./datasets/RDS-ROBO/test/images/",save=True)
#model("./datasets/gdansk_bay/",save=True, box_fusion = 'soft_nms')