# Segmentation-with-NN

### The project

This project consists on an implementation of an encoder-decoder neural network to perform semantic segmentation (every object similar will have the same label, instead of being treated as different instances which is instance segmentation) with deep learning. The encoder minimizes the inputs with Maxpooling to extract most information and Upsampling rebuilds it to make a pixel-wise prediction (every pixel has a label) so the output has the same shape as the input.

An example of encoder-decoder neural network  
![image](https://user-images.githubusercontent.com/91634314/189516462-cb2c77d8-c62c-44f3-b0bb-398532ec5fc1.png)

#### Transforming the dataset

The dataset is composed of mostly landscape with (or without) water as we need to detect it. So there is two label in the output (0,0,0) for water and (255,255,255) for everything else.

For the input, we use OpenCV to load images, resize them into 256*256 shape to make sure we don't loose any dimensions during the Maxpooling phase (the image is resized two times smaller on every Maxpooling and 256 can get divided by 2 until it equals one so it's a good number).
For the output, we do the same thing but also transform the image to a matrix as we put another label on pixels, [1,0] for water and [0,1] for non-water. We then save them as pickle files.

#### Neural network training

First we load the new dataset we made, shuffle it to have some diversity and split it between training (80%) and test (20%). For the validation we take images one by one on google as we don't need so much.

Then we build the model with input (256,256,3) where 3 is the number of color and the output as (256,256,2) as we have 2 classes. The encoder and decoder are symetric and composed of 3 convolutional block with Dropout and BatchNormalization to prevent overfitting.

To train the model, we use Adam optimizer and categorical_crossentropy loss function as we do classification and for the others (batch size, learning rate) I tested some until it was ok. I also found that reducing the learning rate during the training gives us better results so I implemented twon callbacks, one two decrease the lr following an exponential function and one that samples the value of the learning rate. Two other callbacks are here to save the model everytime it gets better (following the validation accuracy) and stop the training if the model doesn't get better after a number of epochs which we defined as 20.

Accuracy plot for the traning (blue is training and orange is validation)  
![Accuracy](https://user-images.githubusercontent.com/91634314/189517925-f07f9717-331b-464f-9529-4fcfcdbf25ec.PNG)

Loss plot for the training  
![Loss](https://user-images.githubusercontent.com/91634314/189517983-af2edef8-1906-43e4-a638-3c1d7b81f47b.PNG)

### Conclusion

At this stage, the model is already really good as we have up to 80% accuracy and as we test it (see Results) we can clearly distinguish the shape of water around objets. Obviously the predictions are not perfect but one limit of the model is that we take a picture of the everyday life (not a landscape) the reflection on water makes it difficult to predict it.

### Results

Input  
![test1bis](https://user-images.githubusercontent.com/91634314/189518741-26a79fe6-f831-4c43-bb7a-e029117e9598.png)

Ground Truth  
![output1](https://user-images.githubusercontent.com/91634314/189516071-6117de08-aecd-400e-89e8-72569fb59711.png)

Prediction  
![output1_pred](https://user-images.githubusercontent.com/91634314/189516080-875a8e81-5104-4a1f-8ae1-5cd3a5abb3a0.PNG)


### Documentation

dataset: https://www.kaggle.com/datasets/gvclsu/water-segmentation-dataset
