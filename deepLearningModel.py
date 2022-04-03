from contextlib import redirect_stdout
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import pandas
import os
import shutil

image_size = (224, 224)
batch_size = 32
epochs = 10

datagen = ImageDataGenerator(
        rotation_range=5,
        rescale=1./255,
        horizontal_flip=True,
        fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

def getModel():
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3)) 
    
    # Freeze convolutional layers
    for layer in base_model.layers:
        layer.trainable = False    

    x = base_model.output
    x = Flatten()(x)  # flatten from convolution tensor output  
    x = Dense(500, activation='relu')(x) # number of layers and units are hyperparameters, as usual
    x = Dense(500, activation='relu')(x)
    predictions = Dense(13, activation='softmax')(x) # should match # of classes predicted

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model

def getTestGen(testLocation):
    test_gen = test_datagen.flow_from_directory(
        testLocation,
        target_size = image_size,
        batch_size = batch_size,
        class_mode = 'categorical',
        color_mode = 'rgb',
        shuffle=False 
    )

    return test_gen


def getTrainGen(trainLocation):
    train_gen = datagen.flow_from_directory(
        trainLocation,
        target_size = image_size,
        batch_size = batch_size,
        class_mode = 'categorical',
        color_mode = 'rgb',
        shuffle=True
    )

    return train_gen

def trainModel(model, trainLoc, testLoc, RunName =""):
    dataDir = "./RunData/" + RunName + "/"
    try:
        os.mkdir(dataDir)
    except FileExistsError:
        print(f"Choose a different run name, {RunName} already has data")
        exit(1)

    shutil.copy("./Data/data_setup.json", dataDir + "model_params.dat")

    with open(dataDir + "model_params.dat", "a") as f:
        print(f"\n\n\nBatch Size:{batch_size}\n, Epochs:{epochs}", file=f)
        with redirect_stdout(f):
            model.summary()

    train_gen = getTrainGen(trainLoc)
    test_gen = getTestGen(testLoc)

    history = model.fit(
        train_gen,
        epochs=epochs,
        verbose =1,
        validation_data= test_gen
    )

    model.save_weights(dataDir + "/model_weights.h5")

    hist_data = pandas.DataFrame(history.history)

    with open(dataDir + "model_history.dat", "w") as historyFile:
        hist_data.to_json(historyFile)

    historyFile.close()

