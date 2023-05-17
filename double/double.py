from keras import *


model = Sequential()

model.add(layers.Dense(units=10, input_shape=[1]))

model.add(layers.Dense(units=64))

model.add(layers.Dense(units=1))

input = [1, 2, 3, 4, 5]
output = [2, 4, 6, 8, 10]

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(x=input, y=output, epochs=1000)


while True :
    x= int(input('Number :'))
    print('Result : ' + str(model.predict([x])))