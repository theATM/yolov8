from ultralytics import YOLO
import os
print(os.getcwd())

# Load a model
model_path = "./pretrained/yolov8s.pt" #"./runs/detect/train14/weights/best.pt" # "yolov8n.pt"
model = YOLO(model_path) # load a pretrained model (recommended for training)
# Use the model
#model.train(data="./datasets/RSD-YOLO/rds-yolo.yaml", epochs=200, batch=8, freeze=10,coco_annot='./datasets/RSD-YOLO/valid/instances_val.json',save_json=True, box_fusion = 'cp_clustering')  # train the model
#model.train(data="./datasets/DOTARO-YOLO/dotaro-yolo.yaml", epochs=200, batch=2, freeze=10,coco_annot='./datasets/DOTARO-YOLO/valid/instances_val.json',save_json=True, box_fusion = 'soft-nms')  # train the model
#model.train(data="./datasets/DOTANA-YOLO/dotana-yolo.yaml", epochs=200, batch=2, freeze=10,coco_annot='./datasets/DOTANA-YOLO/valid/instances_val.json',save_json=True, box_fusion = 'soft-nms')  # train the model
#model.train(data="./datasets/DOTA-YOLO/dota-yolo.yaml", epochs=200, batch=2, freeze=10,coco_annot='./datasets/DOTA-YOLO/valid/instances_val.json',save_json=True, box_fusion = 'cp_clustering')  # train the model
#model.train(data="./datasets/RSD-YOLO-DOTANA/rds-yolo-dotana.yaml", epochs=200, batch=2, freeze=10,coco_annot='./datasets/RSD-YOLO-DOTANA/valid/instances_val.json',save_json=True, box_fusion = 'cp_clustering')  # train the model
#model.train(data="./datasets/RSD-YOLO-DOTANA-CL/rds-yolo-dotana-cl.yaml", epochs=200, batch=2, freeze=10,coco_annot='./datasets/RSD-YOLO-DOTANA-CL/valid/instances_val.json',save_json=True, box_fusion = 'soft-nms')  # train the model
model.train(data="./datasets/DOTANA-YOLO-CL/dotana-yolo-cl.yaml", epochs=200, batch=2, freeze=10,coco_annot='./datasets/DOTANA-YOLO-CL/valid/instances_val.json',save_json=True, box_fusion = 'soft-nms')  # train the model


#model.trainer.validator.metrics.metric <- for metrics
#model.trainer.validator.metrics.metric
results = model.val()  # evaluate model performance on the validation set
print(results) # TODO : P5789
#model("./datasets/RSD-GOD-TXT/images/test",save=True)  # predict on an image
# model.trainer.validator.metrics.metric.map
# CLI:
# yolo detect predict model=yolov8n.pt source="https://ultralytics.com/images/bus.jpg" save=true
# yolo predict model="./runs/detect/train14/weights/best.pt" source="./airbase_8749.jpg" save=true
#  yolo predict model="./runs/detect/train14/weights/best.pt" source="datasets/RSD-GOD-TXT/images/test"