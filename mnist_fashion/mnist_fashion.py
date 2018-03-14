import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras import backend as K

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Reshape data to fit into model
# Images are 28x28 pixels in the dataset
if K.image_data_format() == 'channels_first':
    train_images = x_train.reshape(x_train.shape[0],1,28,28)
    test_images = x_test.reshape(x_test.shape[0],1,28,28)
    input_shape = (1,28,28)
else:
    train_images = x_train.reshape(x_train.shape[0],28,28,1)
    test_images = x_test.reshape(x_test.shape[0],28,28,1)
    input_shape = (28,28,1)
    
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images /= 255
test_images /= 255

# Categorical One Hot Encoding
train_labels = keras.utils.to_categorical(y_train,10)
test_labels = keras.utils.to_categorical(y_test,10)

# To take a look like how the images look like in the dataset
#import matplotlib.pyplot as plt
#def display_sample(index):
#    print(train_labels[index])
#    label = train_labels[index].argmax(axis=0)
#    image = train_images[index].reshape([28,28])
#    plt.title('Sample %d Label %d' % (index,label))
#    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
#    plt.show()
#    
#display_sample(45)  

# Neural Network
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
history = model.fit(train_images, train_labels, batch_size=32,epochs =10, validation_data=(test_images,test_labels))

score = model.evaluate(test_images,test_labels)
print("Test Loss: ",score[0])
print('Test Accuracy: ',score[1])
