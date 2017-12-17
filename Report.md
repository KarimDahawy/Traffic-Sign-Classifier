# **Traffic Sign Classifier** 
------------------------------------

## Traffic Sign Classifier Project
----------------------------------

**The goals of this project are the following:**

    1. Load the data set.

    2. Explore, summarize and visualize the data set.

    3. Design, train and test a model architecture.

    4. Use the model to make predictions on new images.

    5. Analyze the softmax probabilities of the new images.

[//]: # (Image References)

[image1]: ./examples/1.Visualization_of_the_dataset.png
[image2]: ./examples/2.Training_Data.png
[image3]: ./examples/3.Validation_Data.png
[image4]: ./examples/4.Testing_Data.png
[image5]: ./examples/5.Preprocess_the_Training_Data.png
[image6]: ./examples/6.Preprocess_the_Validation_Data.png
[image7]: ./examples/7.Preprocess_the_Testing_Data.png
[image8]: ./examples/8.Training_Accuracy_Vs_Validation_Accuracy.png
[image9]: ./examples/9.New_Loaded_images.png
[image10]: ./examples/10.Preprocessed_new_loaded_images.png
[image11]: ./examples/11.Top_5_Softmax_Probabilities.png
[image12]: ./examples/12.Plot_Top_5_Softmax_Probabilities.png

### 1. Load the Data
---------------------
I have seprated the training, validation and testing data set in 3 different folders and then loaded them using pickle python library in the following train,valid and test. then I separate the data into features and labels.

### 2. Data Set Summary & Exploration
--------------------------------------

**Basic summary of the data set:**

I used the pandas library and python to calculate summary statistics of the traffic
signs data set:

   1. The size of training set is **34799 samples**
   2. The size of the validation set is **4410 samples**
   3. The size of test set is **12630 samples**
   4. The shape of a traffic sign image is **(32, 32, 3)**
   5. The number of unique classes/labels in the data set is **43**


**Visualization of the dataset:**

Here is an exploratory visualization of the data set. I have plotted a sample of each image to identify what each label refers to:

![alt text][image1]

Here's a bar chart showing the count of each image in the training set Data:

![alt text][image2]

Here's a bar chart showing the count of each image in the Validation set Data:

![alt text][image3]

Here's a bar chart showing the count of each image in the Testing set Data:

![alt text][image4]


### 3. Design and Test a Model Architecture
--------------------------------------------

**Pre-process the Data set:**

As a first step, I decided to convert the images to grayscale, So Color won't be a factor in identifying the image and will help in pre processing the images in a easier way

Here is an example of a Training Data set after grayscaling.

![alt text][image5]

Here is an example of a Validation Data set after grayscaling.

![alt text][image6]

Here is an example of a Testing Data set after grayscaling.

![alt text][image7]

As a last step, I normalize all the data sets (Training,Validation and Testing), So all the data sets will have zero mean and equal standard deviation.



**Model Architecture:**

As LeNet is a recommended Model for identifying traffic signs, I have used a modified version of LeNet in my Neural Network to achieve the required validation accuracy.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 GrayScale image   							| 
| Convolution 5x5     	| 1x1 stride, Valid padding, outputs 28x28x45 	|
| RELU					|	Activation Function											|
| Max pooling	      	| 2x2 stride,  outputs 14x14x45 				|
| Convolution 5x5     	| 1x1 stride, Valid padding, outputs 10x10x90 	|
| RELU					|	Activation Function											|
| Max pooling	      	| 2x2 stride,  outputs 5x5x90 				|
| Convolution 3x3     	| 1x1 stride, Valid padding, outputs 3x3x270 	|
| RELU					|	Activation Function											|
| Max pooling	      	| 2x2 stride,  outputs 2x2x270 				|
| Flatten	    | Flatten layer3 (2x2x270) ----> 1080      									|
| Dropout		| Keep probability = 0.5        									|
| Fully Connected				| Input = 1080, Output = 120        									|
| RELU					|	Activation Function											|
| Dropout		| Keep probability = 0.5        									|
| Fully Connected				| Input = 120, Output = 84        									|
| RELU					|	Activation Function											|
| Dropout		| Keep probability = 0.5        									|
| Fully Connected				| Input = 84, Output = 43        									|


**Train the Model**

I have used the following parameters in order to train my model:

1. Arguments used for tf.truncated_normal to define weights:
   * **mu = 0**
   * **sigma = 0.1**
2. Adam Optimizer with Learning rate = **0.0005**
3. Number Of EPOCHS = **20**
4. Batch Size = **128**
5. Dropout with keep_prob = **0.5**


**Model Solution Approach**

First I have used LeNet architecture in order to build my pipeline, but as the validation accuracy given by LeNet was below 93%, I have made some modifications to LeNet so the validation accracy will be Higher.

I have added a new convoulation layer and increased the number of filters in the first layer so I can extract as much feature as I can. then after concatenating the data after the three convoulation layers I have added some dropouts in order to decrease the number of features and avoid overfitting.

My final model results were:

* training set accuracy of **99.9%**
* validation set accuracy of **98%** 
* test set accuracy of **95.5%**

A Chart to identify the training accuracy versus validation accuracy
![alt text][image8]

### 4. Test a Model on New Images:
----------------------------------

**New Images for German Traffic Signs**

![alt text][image9]

I have loaded 21 images in order to fully test the model for different types of signs.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (20km/h)      		| Speed limit (20km/h)  									| 
| Right-of-way at the next intersection     			| Right-of-way at the next intersection										|
| Priority road					| Priority road											|
| Stop	      		| Stop					 				|
| No entry			| No entry      							|
| Dangerous curve to the left			| Dangerous curve to the left      							|
| Speed limit (50km/h)			| Speed limit (50km/h)      							|
| Bumpy road			| Bumpy road      							|
| Road work		| Road work     							|
| Children crossing			| Children crossing      							|
| Bicycles crossing			| Bicycles crossing      							|
| Speed limit (60km/h)			| Speed limit (60km/h)      							|
| Ahead only			| Ahead only     							|
| Go straight or right			| Go straight or right      							|
| Go straight or left			| Go straight or left      							|
| Keep right			| Keep right      							|
| Roundabout mandatory			| Roundabout mandatory      							|
| Speed limit (80km/h)			| Speed limit (80km/h)      							|
| **End of speed limit (80km/h)**			| **Speed limitt (30km/h)**      							|
| **Speed limit (100km/h)**			| **Speed limit (30km/h)**      							|
| No passing			| No passing      							|


The model was able to correctly guess 19 of the 21 traffic signs, which gives an accuracy of 90.476%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

![alt text][image11]

![alt text][image12]

