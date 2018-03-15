from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.preprocessing import sequence

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=20000)

# Add maxlen to reduce computational time and effort. May affect result.
x_train = sequence.pad_sequences(x_train)
x_test = sequence.pad_sequences(x_test)

model = Sequential()
model.add(Embedding(20000,128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x_train,y_train, batch_size=32,epochs=20,validation_data=(x_test,y_test))

score, acc = model.evaluate(x_test, y_test,
                            batch_size=32,
                            verbose=2)
print('Test score:', score)
print('Test accuracy:', acc)
