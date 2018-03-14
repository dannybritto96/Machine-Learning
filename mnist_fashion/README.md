# Convolutional Neural Networks

### Dataset - MNIST Fashion database of fashion articles
[https://keras.io/datasets/]

Dataset of 60,000 28x28 grayscale images of 10 fashion categories, along with a test set of 10,000 images. 
This dataset can be used as a drop-in replacement for MNIST. The class labels are:

|Label  |Description  |
|-------|-------------|
|0      |T-shirt/top  |
|1 	    |Trouser      |
|2      |Pullover     |
|3      |Dress        |
|4 	    |Coat         |
|5 	    |Sandal       |
|6 	    |Shirt        |
|7 	    |Sneaker      |
|8 	    |Bag          |
|9 	    |Ankle boot   |

Convolutional Neural Networks was used for this program. The 28x28 images were reshaped to fit to the model.
2D Convolutional Layer is used in the Neural network. 3x3 convolutional kernel is used.
The first layer is a 2DConv layer in which the input_shape is defined.
MaxPooling2D layer accepts the spatial result of 2DConv layer.
Flatten flattens the input to the Dense layer.

Dense layers are your regular densly connected neural networks.
Dropout is done to avoid overfitting.

#### Accuracy of 92% was acheived.
