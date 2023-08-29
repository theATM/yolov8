from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s.pt")
#model = YOLO("runs/detect/train14/weights/best.pt") #YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model:

#%%
# train the model:
# On Pure RSD
#results = model.train(data="rds-robo.yaml", epochs=200,batch=8,freeze=10,coco_annot='./datasets/RDS-ROBO/valid/instances_val.json', save_json=True, cp_clustering=True)
# On RSD + DOTA:
results = model.train(data="./datasets/RSD-YOLO/rds-yolo.yaml", epochs=100,batch=8,freeze=0,coco_annot='./datasets/RSD-YOLO/valid/instances_val.json',
                      save_json=True, box_fusion='nms',cos_lr=True)

# evaluate model performance on the validation set
results = model.val(save_json=True,plots=True)

# evaluate model performance on the test set
results = model.val(data="./datasets/RSD-YOLO/rds-yolo.yaml",save_json=True,plots=True,split='test',coco_annot='./datasets/RSD-YOLO/test/instances_test.json')


# predict on test images
#results = model("./datasets/RSD-YOLO/test/images/",data="./datasets/RSD-YOLO/rds-yolo.yaml",save=True)

#%%