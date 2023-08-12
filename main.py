from ultralytics import YOLO
from time import perf_counter, strftime
######################################### Main Training Script #######################################################

# Insert dataset path (the yaml file):
DATASET_YAML_PATH = "../datasets/RSD-YOLO/rds-yolo.yaml"

# Choose the model (config.yaml or pretrained weights.pt or a checkpoint.pt):
MODEL_CONFIG = 'yolov8s.yaml' # "yolov8n.pt"

### HYPERPARAMETERS:
EPOCHS = 100
BATCH_SIZE = 8
COSINE_LEARNING_RATE = True

##########
# choose what you want to do:
TRAIN = True # starts a trining process
EVAL =  True # do the model evaluation on val and test sets
PREDICT = False # Predict on the test images (interference on whole dataset takes time)

###################
# Measure time
start_time = perf_counter()

# Load a model
model = YOLO(MODEL_CONFIG)

# Train the model:
if TRAIN:
    results = model.train(data=DATASET_YAML_PATH, epochs=EPOCHS,batch=BATCH_SIZE,save_json=True, cos_lr=COSINE_LEARNING_RATE)

if EVAL:
    # evaluate model performance on the validation set
    results = model.val(save_json=True,plots=True)

    # evaluate model performance on the test set
    results = model.val(data=DATASET_YAML_PATH,save_json=True,plots=True,split='test')

# predict on test images
if PREDICT:
    results = model("./datasets/RSD-YOLO-DOTANA-T/test/images/",data="./datasets/RSD-YOLO-DOTANA-T/rds-yolo-dotana-t.yaml",save=True)

# Measure time
end_time = strftime("%H:%M:%S", perf_counter() - start_time)
print(f"Yolo Script concluded without a fuss. It took {end_time}. Have a nice day! 😄")


#%%