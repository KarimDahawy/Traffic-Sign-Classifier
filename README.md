# **Traffic Sign Classifier** 
------------------------------------
## Overview
-----------
The Purpose of this project is to build a deep neural network based on a modified version of Lenet architecture that will be able to recognize German Traffic Signs.

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


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


