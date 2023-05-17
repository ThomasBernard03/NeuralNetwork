from keras import *

model = Sequential() # Initialize a Sequential model which is a linear stack of layers

# Add a densely-connected layer with 10 units to the model, specifying the input shape to be [1]
# This is the input layer where each neuron (or node) receives input from all the neurons in the previous layer
model.add(layers.Dense(units=10, input_shape=[1]))

model.add(layers.Dense(units=64)) # Add another dense layer with 64 units, this is the hidden layer

model.add(layers.Dense(units=1)) # Add a dense output layer with 1 unit

# Define the input and output lists for training the model
inputList = [1, 2, 3, 4, 5]
outputList = [2, 4, 6, 8, 10]

# Compile the model with Mean Squared Error loss function and Adam optimization algorithm
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model for a defined number of epochs (iterations on a dataset)
model.fit(x=inputList, y=outputList, epochs=1000)

# After training, the model is ready to make predictions
# It will continually ask for a number, make a prediction, and print the result
while True :
    x= int(input('Number :')) # User input
    print('Result : ' + str(model.predict([x]))) # Model prediction and print result
