# Keras model with the same architecture as that in "Make Your Own Neural Network" by T. Rashid

import numpy as np
# scipy.special for the sigmoid function expit()
import scipy.special
# to measure elapsed time
from timeit import default_timer as timer
# pandas for reading CSV files
import pandas as pd
# Keras
from tensorflow import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.utils import np_utils
# suppress Tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def run(learningRate, batchSize, epochNumber, Adam):
    name = f'{"Adam" if Adam else "Ftrl"}'
    print(f"Run {name} - Learning Rate:{learningRate}, Batch:{batchSize}, Epoch:{epochNumber}")
    # start the timer
    start_t = timer()
    # number of input, hidden and output nodes
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10

    # learning rate
    learning_rate = learningRate

    # create a Keras model
    model = Sequential()
    model.add(Dense(hidden_nodes, activation='sigmoid', input_shape=(input_nodes,), use_bias=False))
    model.add(Dense(output_nodes, activation='sigmoid', use_bias=False))
    # print the model summary
    model.summary()
    # set the optimizer (Adam is one of many optimization algorithms derived from stochastic gradient descent)
    if Adam:
        opt  = keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        opt  = keras.optimizers.Ftrl(learning_rate=learning_rate)

    # define the error criterion ("loss"), optimizer and an optional metric to monitor during training
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # load the mnist training data CSV file using pandas
    # note - this CSV file has no headers (i.e., data starts from the first row)
    df = pd.read_csv("../mnist_train.csv", header=None)
    # columns 1-784 are the input values
    x_train = np.asfarray(df.loc[:, 1:input_nodes].values)
    x_train /= 255.0
    # column 0 is the desired label
    labels = df.loc[:,0].values
    # convert labels to one-hot vectors (0, ..., 0, 1, 0, ..., 0)
    y_train = np_utils.to_categorical(labels, output_nodes)

    # train the neural network
    # epochs is the number of times the training data set is used for training
    epochs = epochNumber

    # batch size = 1 to match the previous approach
    batch_size = batchSize

    # train the model
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

    # save the model
    model.save(f'MNIST_3layer_keras_{learning_rate}-{batchSize}-{epochNumber}.h5')
    print('model saved')

    # test the model
    # load the mnist test data CSV file into a list
    #test_data_file = open("mnist_csv/mnist_test_10.csv", 'r')
    test_data_file = open("../mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # scorecard for how well the network performs, initially empty
    scorecard = []

    # go through all the data in the test data set, one by one
    for record in test_data_list:
        # split the record by the ',' commas
        data_sample = record.split(',')
        # correct answer is first value
        correct_label = int(data_sample[0])
        # scale and shift the inputs
        inputs = np.asfarray(data_sample[1:]) / 255.0
    
        # make prediction
        outputs = model.predict(np.reshape(inputs, (1, len(inputs))))

        # the index of the highest value corresponds to the label
        label = np.argmax(outputs)
        # append correct or incorrect to list
        if (label == correct_label):
            # network's answer matches correct answer, add 1 to scorecard
            scorecard.append(1)
        else:
            # network's answer doesn't match correct answer, add 0 to scorecard
            scorecard.append(0)


    # calculate the accuracy, the fraction of correct answers
    scorecard_array = np.asarray(scorecard)
    print("accuracy = {}".format(scorecard_array.sum() / scorecard_array.size))  

    # stop the timer
    end_t = timer()
    print("elapsed time = {} seconds".format(end_t-start_t))

if __name__ == "__main__":
    defaultLearningRate = 0.001
    defaultBatchSize = 32
    defaultNumberEpochs = 5
    Adam = True
    for lr in [0.1, 0.01, 0.001, 0.001]:
        run(lr, defaultBatchSize, defaultNumberEpochs, Adam)
        run(lr, defaultBatchSize, defaultNumberEpochs, not Adam)

    for batch in [1, 4, 16, 64]:
        run(defaultLearningRate, batch, defaultNumberEpochs, Adam)
        run(defaultLearningRate, batch, defaultNumberEpochs, not Adam)

    for epoch in [5, 10, 15, 20]:
        run(defaultLearningRate, defaultBatchSize, epoch, Adam)
        run(defaultLearningRate, defaultBatchSize, epoch, not Adam)
