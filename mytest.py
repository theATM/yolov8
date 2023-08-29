from ultralytics import YOLO

# Load a model from checkpoint
model = YOLO("runs/detect/train9/weights/best.pt")

# evaluate model performance on the test set
#results = model.val(data="./datasets/RSD-YOLO-DOTANA-T/rds-yolo-dotana-t.yaml",
#                    save_json=True,plots=True,split='test',coco_annot='./datasets/RSD-YOLO-DOTANA-T/test/instances_test.json')

# predict on test images
#results = model("./datasets/RSD-YOLO-DOTANA-T/test/images/",data="./datasets/RSD-YOLO-DOTANA-T/rds-yolo-dotana-t.yaml",save=True)
model("./datasets/gdansk_bay/",save=True, box_fusion = 'soft-nms')