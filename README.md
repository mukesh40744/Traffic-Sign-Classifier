**Traffic Sign Classification Project**

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

<table style="width:100%">
  <tr>
    <th>
      <p align="center">
           <img src="./example/SummaryOfDataSet.PNG" alt="summary statistics" width="60%" height="60%">
           <br>Basic summary of the data set
      </p>
    </th>
  </tr>
</table>


#### 2. Include an exploratory visualization of the dataset.

Visualize the German Traffic Signs Dataset using the pickled file(s).

<table style="width:100%">
  <tr>
    <th>
      <p align="center">
           <img src="./example/VisualizationOfDataset.PNG" alt="visualization of the dataset" width="60%" height="60%">
           <br>Visualization of the dataset
      </p>
    </th>
  </tr>
</table>



Below image show the class  distributed in both training and testing set

<table style="width:100%">
  <tr>
    <th>
      <p align="center">
           <img src="./example/ClassDistribution.PNG" alt="class distribution" width="60%" height="60%">
           <br>Class Distribution
      </p>
    </th>
  </tr>
</table>

we see that the data distribution is almost the same between training and testing set.
looks like we won't have problem related to dataset shift when we'll evaluate our model on the test data.

Code for data distibution 

<table style="width:100%">
  <tr>
    <th>
      <p align="center">
           <img src="./example/DataDistibutionCode.PNG" alt="class distribution code" width="60%" height="60%">
           <br>Class Distribution Code
      </p>
    </th>
  </tr>
</table>

### Design and Test a Model Architecture

Below code snippet show the feature extraction -

<table style="width:100%">
  <tr>
    <th>
      <p align="center">
           <img src="./example/FeatureProcessing.PNG" alt="Feature Processing" width="60%" height="60%">
           <br>Feature Processing
      </p>
    </th>
  </tr>
</table>
 
	Below explaintion for the above code-
		1-first i converted the image fomr RGB to YUV
		2-adjuted the image contrast
		3-then extracted the feature for all the train and test data 

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Below are the point i have consider for preprocessed the image data-

1) Each image is converted from RGB to YUV, then only the Y channel is used.

2)Contrast of each image is adjusted by means of histogram equalization. 

3)Each image is centered on zero mean and divided for its standard deviation. This feature scaling is known to have beneficial effects on the gradient descent performed by the optimizer.

<table style="width:100%">
  <tr>
    <th>
      <p align="center">
           <img src="./example/RGBImage.PNG" alt="Feature Processing" width="60%" height="60%">
           <br>Feature Processing
      </p>
    </th>
  </tr>
</table>

 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

To add the additional data, I iterated through each class . The actual additional data was generated from the already existing pictures in each class, by having the loop pick a random angle to rotate using scipy's 'ndimage.rotate' function. I had it go through the loop until enough of these additional pictures were generated to rise to the mean.

 I also shuffled the data just before the split so that I wouldn't end up putting all of a few signs into the validation set, with none in the training set (which would get the model really good at the training set but unable to do anything on the test set).

I will just be using the original test data as the test set.

1- 	Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6. Weight and bias If not using grayscale, the third number in shape would be 3		
			Example -
			c1_weight = tf.Variable(tf.truncated_normal(shape = (5, 5, 1, 6), mean = mu, stddev = sigma))
			c1_bias = tf.Variable(tf.zeros(6))
			conv_layer1 = tf.nn.conv2d(x, c1_weight, strides=[1, 1, 1, 1], padding='VALID') + c1_bias
			conv_layer1 = tf.nn.relu(conv_layer1)
			conv_layer1 = tf.nn.avg_pool(conv_layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
			
2- Layer 2: Convolutional. Output = 10x10x16
			c2_weight = tf.Variable(tf.truncated_normal(shape = (5, 5, 6, 16), mean = mu, stddev = sigma))
			c2_bias = tf.Variable(tf.zeros(16))
			conv_layer2 = tf.nn.conv2d(conv_layer1, c2_weight, strides=[1, 1, 1, 1], padding='VALID') + c2_bias
			conv_layer2 = tf.nn.relu(conv_layer2)
			conv_layer2 = tf.nn.avg_pool(conv_layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
			
3- Layer 3:	Fully Connected. Input = 400. Output = 120. Although this is fully connected, the weights and biases still are implemented similarly 
			There is no filter this time, so shape only takes input and output
			
			fc1_weight = tf.Variable(tf.truncated_normal(shape = (400, 200), mean = mu, stddev = sigma))
			fc1_bias = tf.Variable(tf.zeros(200))
			fc1 = tf.matmul(flat, fc1_weight) + fc1_bias
			fc1 = tf.nn.relu(fc1)
			fc1 = tf.nn.dropout(fc1, keep_prob)	
4- Layer 4 : Fully Connected. Input = 120. Output = 84. Same as the fc1 layer, just with updated output numbers		
			
			fc2_weight = tf.Variable(tf.truncated_normal(shape = (200, 100), mean = mu, stddev = sigma))
			fc2_bias = tf.Variable(tf.zeros(100))
			fc2 = tf.matmul(fc1, fc2_weight) + fc2_bias
			fc2 = tf.nn.relu(fc2)
			fc2 = tf.nn.dropout(fc2, keep_prob)|	
5- Layer 5 : Fully Connected. Input = 84. Output = 43	
		
			fc3_weight = tf.Variable(tf.truncated_normal(shape = (100, 43), mean = mu, stddev = sigma))
			fc3_bias = tf.Variable(tf.zeros(43))
			logits = tf.matmul(fc2, fc3_weight) + fc3_bias
 

Below is the code of placeholder -

x = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 1))
y = tf.placeholder(dtype=tf.int32, shape=None)
keep_prob = tf.placeholder(tf.float32)

Below is the code for training model

# training pipeline
lr = 0.001
logits ,conv1, conv2= neuralNetwork(x)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
loss_function = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_step = optimizer.minimize(loss=loss_function)


# metrics and functions for model evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.cast(y, tf.int64))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

		Traning the model I used Adam optimizer. 
		Batchsize is set to 128 . Every 5000 batches visited, an evaluation on both training and validation set is performed. 
		In order to avoid overfitting, both data augmentation and dropout (with drop probability of 0.5) are employed extensively.
		
		
#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The network architecture is based on the paper [Sermanet, LeCun], in which the authors tackle the same problem (traffic sign classification), 
though using a different dataset. In section II-A of the paper, the authors explain that they found beneficial to feed the dense layers with the 
output of both the previous convolutional layers. Indeed, in this way the classifier is explicitly provided both the local "motifs"  
and the more "global" shapes and structure found in the features. I tried to replicate the same architecture, made by 2 
convolutional and 3 fully connected layers. The number of features learned was lowered until the training was feasible also in my laptop!

<table style="width:100%">
  <tr>
    <th>
      <p align="center">
           <img src="./example/PerformanceOnTestSet.PNG" alt="Feature Processing" width="60%" height="60%">
           <br>Feature Processing
      </p>
    </th>
  </tr>
</table>
 
 
This network is a CNN. I mostly used the same architecture as the LeNet neural network did, with 2 convolutional layers and 3 fully connected layers. I also did a few attempts with one less convolutional layer  as well as one less fully connected layer (which only marginally dropped the accuracy).

One item I did change from the basic LeNet structure was adding dropout to the fully connected layers. Although this makes initial epochs in validation a little worse, I gained an additional 3% on test accuracy. Since I was getting to validation accuracy of around 97%, with test accuracy down by 88-89%, there was clearly a little bit of overfitting being done. Dropout helped get my test accuracy into the 90's by preventing some of that overfitting. 

Final Train Accuracy = 0.992 - Validation Accuracy: 0.985

Below images show the training and validation accuracy leranign path


<table style="width:100%">
  <tr>
    <th>
      <p align="center">
           <img src="./example/TrainAccuracy.PNG" alt="Train Accuracy" width="60%" height="60%">
           <br>Train Accuracy
      </p>
    </th>
  </tr>
</table>


The results of this are shown below; not that this is not a true grid search as I was only tuning one parameter at a time as opposed to checking every combination of the below items. For speed's sake, I stuck to 10 epochs, although there is definitely a potential had I ran through something like 100 epochs to improve the validation accuracy while at the same time arriving at a better final test score. The default otherwise used was a learning rate of .01, 150 batch size, 2 convolutional layers and 3 fully connected layers (which is after a little bit of guess and check already).

Learning rate after 30 epochs:

Final Train Accuracy = 0.992 - Validation Accuracy: 0.985



### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

German signs I got were of 32x32x3

Below are image i have used for testing the model-

<table style="width:100%">
  <tr>
    <th>
      <p align="center">
           <img src="./example/NewImagesForTesting.PNG" alt="Feature Processing" width="60%" height="60%">
           <br>Feature Processing
      </p>
    </th>
  </tr>
</table>

From the training and testing set we have seen that contrast of the image could affect the classification. Although I didn't try histogram equalization, 2 of the DeepNet networks use 1x1 filters as the first layer which tend to make the contrast insignificant. The angle of the traffic sign shouldn't be a big problem since our augmented data set already generates images based on random jittering between -10 and 10 degrees. Another thing to note is that all the images were cropped to include only the sign as the most significant object. However, that could pose a problem with images with backgrounds and I didn't get a chance to test those. Some of the images are completely different from what the network has seen and hence it was evident that the none of the networks were able to classify them correctly

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

<table style="width:100%">
  <tr>
    <th>
      <p align="center">
           <img src="./example/NewImageProcessingResult.PNG" alt="NewImage Processing Result" width="60%" height="60%">
           <br>NewImage Processing Result
      </p>
    </th>
  </tr>
</table>

The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 80%. 

Accuracy
Validation Accuracy: ~98%
Testing Accuracy: 91.4%
Real World Accuracy: 3 out of 5 images (~60%)
The reason the network didn't perform well on these images is because signs of these categories were not included in the training set.

Performance on test set: 0.945


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Below image show the perfomrance on new images

<table style="width:100%">
  <tr>
    <th>
      <p align="center">
           <img src="./example/PerformanceOnNewImages.PNG" alt="PerformanceOnNewImages" width="60%" height="60%">
           <br>Performance On New Images
      </p>
    </th>
  </tr>
</table>

Below is the soft max visualization

<table style="width:100%">
  <tr>
    <th>
      <p align="center">
           <img src="./example/SoftMaxVisulaization.PNG" alt="Soft Max Visulaization" width="60%" height="60%">
           <br>Soft Max Visulaization
      </p>
    </th>
  </tr>
</table>

Below is the soft max perfomrance SoftMaxProfarmance

<table style="width:100%">
  <tr>
    <th>
      <p align="center">
           <img src="./example/SoftMaxProfarmance.PNG" alt="Soft Max Profarmance" width="60%" height="60%">
           <br>Soft Max Profarmance
      </p>
    </th>
  </tr>
</table>


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

<table style="width:100%">
  <tr>
    <th>
      <p align="center">
           <img src="./example/NetworkState.PNG" alt="Neural Network State" width="60%" height="60%">
           <br>Neural Network State
      </p>
    </th>
  </tr>
</table>


