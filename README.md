# Yolov5 for Fire Detection
Fire detection task aims to identify fire or flame in a video and put a bounding box around it. This repo includes a demo on how to build a fire detection detector using YOLOv5 and AURDINO UNO. 


<p align="center">
  <img src="results/result.gif" />
</p>


## IOT-DIAGRAM:
In this project Aurdino uno board has been used for obtaining the output. The circut diagram is shown below as
![image](https://user-images.githubusercontent.com/114398468/212620542-1d43ff3b-f32f-4c30-aea5-a2541f61e5ca.png)


## Step by step guide

- Open firedetection.ipynb file in the repository
- In the file clone the repository of yolov5
- Install the requirements.txt
- Divide the dataset(given) into train set and validation set
- Open dataset.yaml file change the directory of source data, train image directory, val image directory
- Copy the file "dataset.yaml" into yolov5->data (yolov5 folder will be created after the cloning)
- Training yolov5 with cuntom fire dataset, Batch: 7, number of epochs: 10, weights: yolov5s.pt
- Testing the model with real time data using web cam. code: !python detect.py --source 0--weights .yolov5/runs/train/exp/weights/best.pt

#### Install
Clone this repo and use the following script to install [YOLOv5](https://github.com/ultralytics/yolov5). 
```

# Install yolov5
git clone https://github.com/ultralytics/yolov5  
cd yolov5
pip install -r requirements.txt
```

#### Training
Set up ```train.ipynb``` script for training the model from scratch. To train the model, download [Fire-Dataset](https://drive.google.com/file/d/1TQKA9nzo0BVwtmojmSusDt5j02KWzIu9/view?usp=sharing) and put it in ```datasets``` folder. This dataset contains samples from both [Fire & Smoke](https://www.kaggle.com/dataclusterlabs/fire-and-smoke-dataset) and [Fire & Guns](https://www.kaggle.com/atulyakumar98/fire-and-gun-dataset) datasets on Kaggle.
```
python train.py --img 640 --batch 16 --epochs 10 --data ../fire_config.yaml --weights yolov5s.pt --workers 0
```
#### Prediction
If you train your own model, use the following command for detection:
```
python detect.py --source ../input.mp4 --weights runs/train/exp/weights/best.pt --conf 0.2
```
Or you can use the pretrained model located in ```models``` folder for detection as follows:
```
python detect.py --source ../input.mp4 --weights ../models/best.pt --conf 0.2
```





