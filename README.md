In this project a mobilenetv2 is trained on a custom dataset consists of classes Birds, Large Quadcopters, Small Qualdcopters and Winged Drones. The training is transfer learning i.e., a pretrained pytorch model is picked up from torchvision.models and further trained on new dataset. For this the last classifier layer is changed to predict output for 4 classes which originally has 10 classes.

## Following are the steps performed in order to preprocess dataset before it to be passed for training-

 1. A very large group has collected dataset consists of aforementioned classes at a shared gdrive location. Download raw dataset from shared gdrive - <link>
 2. Segregation of dataset into train and test is required in order to train and test mobilenetv2 accuracy. The linux command below easily help segregating data of {i} classes into train and test. The command it to be run fro raw data {i} class folder. The command copy images into target/{train/test}/{i}class/ directorires -
	```2.1 find . -maxdepth 1 -type f | head -n70 | xargs -I X cp X <target/train/{i}dir>```
	```2.2 find . -maxdepth 1 -type f | tail -n30 | xargs -I X cp X <target/test/{i}dir>```
	```70:30 is the split for train and test.```
 3. Once the dataset is structured it would appear like this
	```-root/dir
		-train
			-0
			-1
			-2
			-3
		-test
			-0
			-1
			-2
			-3
		words.txt
 4. 0,1,2,3 are the four classes as mentioned in point #1 of this README.md on which our mobilenetv2 is trained. File words.txt is a class to label mapping file. For ease the structure of datatest is kept similar to tinyimagenet dataset.
 5. Cleansing of raw dataset is needed post segregation into train and test as not all the images in the dataset are in JPG format. Use the following linux command to convert PNG and JPEG into JPG. The below command also displays on terminal which all images are not able to convert because the images are corrupt images hence once can easily clean them from the directory-
	```5.1 for i in *.png; do convert "$i" "${i%.*}.jpg" ; done```
	```5.2 for i in *.jpeg; do convert "$i" "${i%.*}.jpg" ; done```
 6. Use the following command to download the dataset when traning from colab
	``` #!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=10EPoE4EuFQ6Sq8VZTQmboIiQLdk1w_gl' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=10EPoE4EuFQ6Sq8VZTQmboIiQLdk1w_gl" -O session2-dataset ```


### Adaptive Maxpooling

In average-pooling or max-pooling, we essentially set the stride and kernel-size by our own, setting them as hyper-parameters. We have to re-configure them if we happen to change the input size.

In Adaptive Pooling on the other hand, we specify the output size instead. And the stride and kernel-size are automatically selected to adapt to the needs. The following equations are used to calculate the value in the source code.

```python
Stride = (input_size//output_size)  
Kernel size = input_size - (output_size-1)*stride  
Padding = 0
```
