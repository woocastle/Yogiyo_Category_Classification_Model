# modeling
import numpy as np      # tensorflow 2.9.1 깔아야함 / pip install tensorflow == 2.9.1
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *

X_train, X_test, Y_train, Y_test = np.load(
    './yogiyo_data_max_5_wordsize_7021.npy', allow_pickle=True)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

model = Sequential()
model.add(Embedding(7021, 150, input_length=5))                 # 7021차원을 150차원으로 줄이는 것
model.add(Conv1D(32, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPool1D(pool_size=1))           # pool_size는 변화 없으므로 안써줘도 되지만 관습적으로 써놓자
model.add(GRU(128, activation='tanh', return_sequences=True))           # return_sequences=True 를 주면 계산해서 나오는 값들을 저장해서 sequences 형태로 넘겨준다.
model.add(Dropout(0.3))
model.add(GRU(64, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(GRU(64, activation='tanh'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(9, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
fit_hist = model.fit(X_train, Y_train, batch_size=64,
                     epochs=100, validation_data=(X_train, Y_train))
model.save('./yogiyo_category_classification_model_{}.h5'.format(
    np.round(fit_hist.history['val_accuracy'][-1], 3)))
plt.plot(fit_hist.history['accuracy'], label='accuracy')
plt.plot(fit_hist.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()