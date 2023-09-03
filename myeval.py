from ultralytics import YOLO

# Load a model
model = YOLO("runs/detect/train6/weights/best.pt") #YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model:


#%%
# evaluate model performance on the validation set
#results = model.val(data="./datasets/RSD-YOLO-DOTANA-T/rds-yolo-dotana-t.yaml", save_json=True,plots=True, box_fusion='soft-nms',coco_annot='./datasets/RSD-YOLO-DOTANA-T/valid/instances_val.json')

# evaluate model performance on the test set
#results = model.val(data="./datasets/RSD-YOLO-DOTANA-T/rds-yolo-dotana-t.yaml",save_json=True,plots=True,split='test',coco_annot='./datasets/RSD-YOLO-DOTANA-T/test/instances_test.json', box_fusion='soft-nms')

# predict on test images
results = model("./datasets/RSD-YOLO-DOTANA-T/test/images/",data="./datasets/RSD-YOLO-DOTANA-T/rds-yolo-dotana-t.yaml",save=True) # stream=True

#%%