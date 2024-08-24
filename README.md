
![yolo_baner](https://github.com/theATM/yolov8/assets/48883111/49206b82-7f24-49eb-ae4b-3e2e7bae4295)

# YOLOv8 for the Remote Sensing Code
This code is part of the Master thesis on Remote Sensing Object Detection [REPO](https://github.com/theATM/AirDetection). <br>
It is a fork of the https://github.com/ultralytics/ultralytics Repository.

Main changes to the original YOLOv8 setup are:
* Usage of Remote Sensing Object Detection Dataset 
<a href="https://github.com/Dr-Zhuang/geospatial-object-detection">RSD-GOD</a>
* Usage of RSD-GOD + <a href="https://captain-whu.github.io/DOTA/dataset.html">DOTA</a>
 hybrid dataset called Dotana
* Experiments with the NMS substitute <a href="https://github.com/shenyi0220/CP-Cluster">CP-Cluster</a>
and SoftNMS (mmcv custom build)
* Experiments with different batch sizes, with frozen layers (yolov5 code) and with different optimizers


Branches:
* Main - Clean Repository compatible with 8.0.112 version, training on the RSD and Dotana datasets done via Docker (dockerfile)
* Original_dev - Fork of the version 8.0.21, added the freezing code and the cp-clustering
* CP_Cluster_dev - Original_dev branch updated to the newer version 8.0.112, with multiple custom training pipelines
* ROCm_dev - Branch used for the ROCm compatible training
* Freeze_old - depricated branch

### Yolov8 Usage
🤖 Run the main.py script 😎

The main.py scipt allows to:
* Train the model on both RSD-GOD and Dotana datasets 🏋️
* Evaluate pretrained models on both eval and test datasets 📏
* Detect objects in the images from chosen folder 🔎


### Yolov5 Training

To run experiments on Yolov5 simply copy the RSD-GOD and Dotana .yaml configuration files to the <a href="https://github.com/ultralytics/yolov5">Yolov5</a> Repo.

Then run train.py with example parameters: 
* train.py --data rsd-god.yaml --weights yolov5s.pt --epochs 100 --batch 16
* train.py --data dotana.yaml --weights yolov5s.pt --epochs 100 --batch 16 --freeze 4

To get COCO styled metricies change the save_json to True in val.py and add own path for the coco_instances_val.json (in yolified coco format)

To validate run val.py --data rsd-god.yaml --weights mymodels/best.pt --save-json

To predict on images run detect.py --weights mymodels/best.pt --source dir_with_images/
  

### Yolov8 Results 🚀

#### mAP score on the RSD-GOD dataset
![image](https://github.com/theATM/yolov8/assets/48883111/f74c301b-ff75-4c0e-8549-6da5ba13482f)

#### Per class results
![image](https://github.com/theATM/yolov8/assets/48883111/997f5c6b-b6c4-4f5e-b9c5-88f9da2971cd)


#### Detection Results of the Best YOLOv8 model
|  Airbase  | Helicopters |Oiltanks|Planes|Warships|
|-|-|-|-|-|
| ![image](https://github.com/theATM/yolov8/assets/48883111/31848fbd-c522-4616-ae26-0e90fba7d66c) | ![image](https://github.com/theATM/yolov8/assets/48883111/18de31f3-86ee-489e-9a4a-d87541413d2d) | ![image](https://github.com/theATM/yolov8/assets/48883111/7f7c777a-11b1-4b6d-b1cc-dc2a367fbb7a) | ![image](https://github.com/theATM/yolov8/assets/48883111/2012d5f7-2567-4adb-af11-890f1ec78d3e) | ![image](https://github.com/theATM/yolov8/assets/48883111/c45b26f6-bb2e-4869-8f3f-1d26cff667a3) |



#### Model Zoo

* SoftNMS Models: [Y28](https://drive.google.com/file/d/1ai4D---5uvQeoz2RzkisL5hlNCXDshSg/view?usp=sharing)  [Y9](https://drive.google.com/file/d/1bk0tnVXpOP7wc_9Pp2L16MuhuW1vxDG2/view?usp=sharing)

* CP-Cluster Model: [Y25](https://drive.google.com/file/d/1I4L0x9Hoo-8R9oGka45giOWHbq33AcEG/view?usp=sharing)

* NMS Model: [Y7](https://drive.google.com/file/d/1g9L0rVkM9B2IeH6rYLcdfWcCwUKSuw_5/view?usp=sharing)

* Yolov8l + SoftNMS Models: [L6](https://drive.google.com/file/d/1PMvREHjc_NFcAvgTdImDsp0ooyG631kQ/view?usp=sharing) [L8](https://drive.google.com/file/d/1eKePU7NxfheCx19Yjb_ijPqgk5n-EY7G/view?usp=sharing) [L31](https://drive.google.com/file/d/1MHUkYqBYJTLESNbcjDCOH0bHE1v8X1Fx/view?usp=sharing)

### Original README [Here](README_ORIGINAL.md)


