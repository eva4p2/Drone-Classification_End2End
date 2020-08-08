# Drone Classification Using MobilenetV2 [End to End]

In this project we are doing end to end deploying of MobilenetV2 model for custom drone dataset consists of four classes - ***Flying Birds, Large Quadcopters, Small Qualdcopters and Winged Drones***. 

The project steps includes ***drone dataset preparations***, ***transfer learning with pytorch*** and ***deploy in aws-lambda*** using serverless and docker. 

This project gave us a proper understanding of the techniques, steps and difficulties to make a end to end deep learning project.

## Overview

### Dataset

Dataset Link : 

Total Images :

Team Contribution : 

### Model Training

* Transfer learning is used to train pre-trained MobilenetV2 model for cutome dataset.
* Test Acuracy : 95.37%

### Deploy

Model deployed in AWS Lambda

POST - https://fowahmw57k.execute-api.ap-south-1.amazonaws.com/dev/classify

![deployed image](output/deploy.png)



## In Detail

### 1. Dataset Preparation

1. A very large group has collected dataset consists of aforementioned classes at a shared gdrive location. Download raw dataset from shared gdrive - <link>
 2. Segregation of dataset into train and test is required in order to train and test mobilenetv2 accuracy. The linux command below easily help segregating data of {i} classes into train and test. The command it to be run fro raw data {i} class folder. The command copy images into target/{train/test}/{i}class/ directorires -
	```2.1 find . -maxdepth 1 -type f | head -n70 | xargs -I X cp X <target/train/{i}dir>```
	```2.2 find . -maxdepth 1 -type f | tail -n30 | xargs -I X cp X <target/test/{i}dir>```
	```70:30 is the split for train and test.```
 3. Once the dataset is structured it would appear like this
	```-root/dir
		-train
			-Flying Birds
			-Large QuadCopters
			-Small QuadCopters
			-Winged Drones
		-test
			-Flying Birds
			-Large QuadCopters
			-Small QuadCopters
			-Winged Drones
	```
 5. Cleansing of raw dataset is needed post segregation into train and test as not all the images in the dataset are in JPG format. Use the following linux command to convert PNG and JPEG into JPG. The below command also displays on terminal which all images are not able to convert because the images are corrupt images hence once can easily clean them from the directory-
	```5.1 for i in *.png; do convert "$i" "${i%.*}.jpg" ; done```
	```5.2 for i in *.jpeg; do convert "$i" "${i%.*}.jpg" ; done```
 6. Use the following command to download the dataset when traning from colab
	``` #!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=10EPoE4EuFQ6Sq8VZTQmboIiQLdk1w_gl' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=10EPoE4EuFQ6Sq8VZTQmboIiQLdk1w_gl" -O session2-dataset ```

###  3. Model Description

We did transfer learning using Mobilenet V2 model which pre-trained with Imagenet.

#### MobilenetV2 Model

The MobileNet v2 architecture is based on an inverted residual structure where the input and output of the residual block are thin bottleneck layers opposite to traditional residual models which use expanded representations in the input. MobileNet v2 uses lightweight depthwise convolutions to filter features in the intermediate expansion layer. Additionally, non-linearities in the narrow layers were removed in order to maintain representational power.

![alt](https://pytorch.org/assets/images/mobilenet_v2_1.png) | ![alt](https://pytorch.org/assets/images/mobilenet_v2_2.png)

***To train our drone dataset which having 4 classes, one more fully-connected layer of size 1x1x4 added at the end of MobilenetV2  network.***

```python
# custom mobileV2 model
class DroneMobilenetV2(DroneClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained MobilenetV2 model
        self.network = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
        # Replace last layer
        num_ftrs = self.network.classifier[1].in_features
        self.network.classifier[1] = nn.Linear(num_ftrs, 4)
    
    def forward(self, xb):
        out     = self.network(xb)
        return F.log_softmax(out, dim=1)
```



### 4. Training



### 5. Mis-Classified Images







