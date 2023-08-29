from ultralytics import YOLO
from time import perf_counter, strftime
######################################### Main Training Script #######################################################

# Insert dataset path (the yaml file):
#DATASET_YAML_PATH = "./datasets/RSD-YOLO/rds-yolo.yaml"
DATASET_YAML_PATH = "./datasets/RSD-YOLO-DOTANA-T/rds-yolo-dotana-t.yaml"

#Predict data path:
PREDICT_DIR =  '/'.join(DATASET_YAML_PATH.split('/')[:-1]) + "/test/images/"

# Choose the model (config.yaml or pretrained weights.pt or a checkpoint.pt):
#MODEL_CONFIG = 'yolov8s.yaml'
MODEL_CONFIG =  "yolov8s.pt"

### HYPERPARAMETERS:
EPOCHS = 100
BATCH_SIZE = 8
COSINE_LEARNING_RATE = True

##########
# choose what you want to do:
TRAIN = True # starts a trining process
EVAL =  True # do the model evaluation on val and test sets
PREDICT = False # Predict on the test images (interference on whole dataset takes time)

# If you want to print coco styled evaluation metrics:
COCO_EVAL = True  #( the dataset needs to have instances_set.json (a yolified coco format annotations))

###################
# Measure time
start_time = perf_counter()

# Load a model
model = YOLO(MODEL_CONFIG)

# Train the model:
if TRAIN:
    coco_annot = '/'.join(DATASET_YAML_PATH.split('/')[:-1]) + '/valid/instances_val.json' if COCO_EVAL else ''
    results = model.train(data=DATASET_YAML_PATH, epochs=EPOCHS,batch=BATCH_SIZE,save_json=True, cos_lr=COSINE_LEARNING_RATE, coco_annot=coco_annot)

if EVAL:
    # evaluate model performance on the validation set
    results = model.val(save_json=True,plots=True)

    # evaluate model performance on the test set
    coco_annot = '/'.join(DATASET_YAML_PATH.split('/')[:-1]) + '/test/instances_test.json' if COCO_EVAL else ''
    results = model.val(data=DATASET_YAML_PATH,save_json=True,plots=True,split='test',coco_annot=coco_annot)

# predict on test images
if PREDICT:
    coco_annot = '/'.join(DATASET_YAML_PATH.split('/')[:-1]) + '/test/instances_test.json' if COCO_EVAL else ''
    results = model(PREDICT_DIR,data=DATASET_YAML_PATH,save=True,coco_annot=coco_annot)

# Measure time
end_time = strftime("%H:%M:%S", perf_counter() - start_time)
print(f"Yolo Script concluded without a fuss. It took {end_time}. Have a nice day! 😄")


#%%