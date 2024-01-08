import random
import statistics as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()#normalize=True, one_hot_label = True

# 각종 파라메터의 영향을 보기 위해 랜덤값 고정
tf.random.set_seed(1234)

# Normalizing data
#a = random.randrange(0)
a = 0

numOfdata = 60000
x_train, x_test = x_train/255.0, x_test / 255.0 #50000개 중에 원하는 만큼 데이터를 가져옴???????


# (60000, 28, 28) => (60000, 28, 28, 1)로 reshape
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# One-hot 인코딩
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

#print(len(x_train))#숫자 배열 랜덤한지 보는거



model0 = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu',
                               input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
model1 = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu',
                               input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
model2 = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu',
                               input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
model3 = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu',
                               input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
model4 = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu',
                               input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
model5 = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu',
                               input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
model6 = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu',
                               input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
model7 = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu',
                               input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
model8 = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu',
                               input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
model9 = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu',
                               input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])



model0.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
#model.summary()

################ 10번학습
x_train_numOfdata = x_train[a:a + numOfdata]  # 49500개 중에 200개 가져옴???????
y_train_numOfdata = y_train[a:a + numOfdata]  # 49500개 중에 200개 가져옴???????
model0.fit(x_train_numOfdata, y_train_numOfdata, batch_size=64, epochs=10, validation_data=(x_test, y_test))
model0.save("model.h5")
result = model0.evaluate(x_test, y_test)
print("최종 예측 성공률(%): ", result[1]*100)


################ 1번학습 10개
for i in range(0,10):
    #a = random.randrange(0, 49500)
    x_train_numOfdata = x_train[a:a + numOfdata]  # 49500개 중에 200개 가져옴???????
    y_train_numOfdata = y_train[a:a + numOfdata]  # 49500개 중에 200개 가져옴???????

    # model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))
    model0.fit(x_train_numOfdata, y_train_numOfdata, batch_size=64, epochs=1, validation_data=(x_test, y_test))
    model0.save("model" + str(i) + ".h5")



model0.load_weights("model0.h5")
model1.load_weights("model1.h5")
model2.load_weights("model2.h5")
model3.load_weights("model3.h5")
model4.load_weights("model4.h5")
model5.load_weights("model5.h5")
model6.load_weights("model6.h5")
model7.load_weights("model7.h5")
model8.load_weights("model8.h5")
model9.load_weights("model9.h5")


model0.get_weights()
model1.get_weights()
model2.get_weights()
model3.get_weights()
model4.get_weights()
model5.get_weights()
model6.get_weights()
model7.get_weights()
model8.get_weights()
model9.get_weights()


weights = [model0.get_weights(), model1.get_weights(), model2.get_weights(), model3.get_weights(), model4.get_weights(), model5.get_weights(),
           model6.get_weights(), model7.get_weights(), model8.get_weights(), model9.get_weights()]

#print(len(weights))
new_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu',
                               input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
new_model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(lr=0.001), metrics=['accuracy'])

new_weights = list()

for weights_list_tuple in zip(*weights):
    new_weights.append(
        np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
    )

new_model.set_weights(new_weights)

result = new_model.evaluate(x_test, y_test)
print("최종 예측 성공률(%): ", result[1]*100)



#model.save("model.h5")

# model.load_weights("model.h5")
# model.get_weights()
# print(model.get_weights())
#
# weights = [model.get_weights()]
# weights = model.load_weights("model.h5")
# st.median(model.load_weights("model.h5"))
# print(st.median(model.load_weights("model.h5")))


#x_train = dataset[n]
#n = 랜던 숫자 할당 함수

# train_size = x_train.shape[0]
# batch_size = 10
# batch_mask = np.random.choice(train_size, batch_size)
#
# x_batch = x_train[batch_mask]
# t_batch = x_train[batch_mask]