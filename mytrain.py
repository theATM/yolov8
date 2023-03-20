from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s.pt") #YOLO("runs/detect/train/weights/best.pt") #YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)


#%%
# Use the model
results = model.train(data="rds-god.yaml", epochs=10,batch=8,freeze=10)  # train the model

#%%
#results = model.val()  # evaluate model performance on the validation set

#%%
#results = model("./datasets/RSD-GOD-TXT/images/test/",save=True)  # predict on an image

#%%
#success = model.export(format="onnx")  # export the model to ONNX format