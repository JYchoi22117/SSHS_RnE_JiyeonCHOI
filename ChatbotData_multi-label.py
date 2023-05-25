import pickle
import random

import keras


def save(data, thread=0):
    with open(FR"C:\Users\jiyeo\Documents\카카오톡 받은 파일\R{thread}.adofai", "wb+") as f: #F->FR (역슬래시) #write binary +는 만든다
        pickle.dump(data, f)

def load(thread=0):
    while True:
        try:
            with open(FR"C:\Users\jiyeo\Documents\카카오톡 받은 파일\R{thread}.adofai", "rb") as f:
                return pickle.load(f)
            
        except:
            ...


sentences = load(1000)
responses = load(1001)
labels = load(1002)

assert len(sentences) == len(responses) == len(labels)
t = [(sentences[i], responses[i], labels[i]) for i in range(len(sentences))]

random.shuffle(t)

sentences = [i[0] for i in t]
responses = [i[1] for i in t]
labels = [i[2] for i in t]

import tensorflow as tf

#N_data=len(sentences)

tokenizer=tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(sentences)

vec_sen=tokenizer.texts_to_sequences(sentences)
vec_res=tokenizer.texts_to_sequences(responses)


# print(vec_sen)
# print(vec_res)
# print(labels)

import numpy as np


def vectorize_sequences(sequences, dimension=100000):
    results = np.zeros((len(sequences), dimension)) #np.zeros(x, y): x*y의 2D 배열
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

# 훈련 데이터 벡터 변환
train_x = vectorize_sequences(vec_sen) #train_data -> vec_sen
# 테스트 데이터 벡터 변환
# test_x = vectorize_sequences(test_data)

print('train_x.shape : {}'.format(train_x.shape))
# print('test_x.shape : {}'.format(test_x.shape))



def to_one_hot(labels, dimension=3): #dim 46 -> 3
    results = np.zeros((len(labels), dimension))
    for idx, label in enumerate(labels):
        results[idx, label] = 1.
    return results

# 훈련 레이블 벡터 변환
one_hot_train_labels = to_one_hot(labels) #train_labels
# 테스트 레이블 벡터 변환
# one_hot_test_labels = to_one_hot(test_labels)


from keras.utils.np_utils import to_categorical

one_hot_train_labels = to_categorical(labels) #train_labels
# one_hot_test_labels = to_categorical(test_labels)

print('one_hot_train_labels.shape :', one_hot_train_labels.shape)
# print('one_hot_test_labels.shape :', one_hot_test_labels.shape)


from keras import backend as K
from keras import layers, models

K.clear_session()

model = models.Sequential(
    [layers.Dense(64, activation='relu', input_shape=(100000,)), 
     layers.Dense(64, activation='relu'),
     layers.Dense(3, activation='softmax')]
)
model.summary()

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])


val_x = train_x[:100]
partial_train_x = train_x[100:]

val_y = one_hot_train_labels[:100]
partial_train_y = one_hot_train_labels[100:]


history = model.fit(partial_train_x,
                    partial_train_y,
                    epochs=10, #20->3 
                    batch_size=128,
                    validation_data=(val_x, val_y))



# %matplotlib inline
import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend();


plt.clf()   # 그래프를 초기화합니다

acc = history.history['accuracy'] #acc -> accuracy
val_acc = history.history['val_accuracy'] # "

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

