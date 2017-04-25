import tensorflow as tf
from keras.models import Sequential
from keras import losses
from keras import metrics
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers import Flatten, Dense, Lambda, Dropout, Conv2D, MaxPooling2D, Cropping2D, LeakyReLU
from keras.applications.inception_v3 import InceptionV3


def get_model(model, args):
    '''Given a model name, return the correct model'''
    if args['model'] == 'nvidia':
        return nvidia_model(model, args)
    elif args['model'] == 'nvidia_conv_relu':
        return nvidia_conv_relu_model(model, args)
    elif args['model'] == 'nvidia_conv_dropout':
        return nvidia_conv_dropout_model(model, args)
    elif args['model'] == 'nvidia_relu':
        return nvidia_relu_model(model, args)        
    elif args['model'] == 'nvidia_relu_nadam_model':
        return nvidia_relu_nadam_model(model, args)    
    elif args['model'] == 'nvidia_lrelu':
        return nvidia_lrelu_model(model, args)       
    elif args['model'] == 'simple_conv_1':
        return simple_conv_1(model, args)      
    elif args['model'] == 'inception_v3_model':
        return inception_v3_model(model, args)      
    elif args['model'] == 'nvidia_lrelu_dropout':
        return nvidia_lrelu_dropout_model(model, args)  
    elif args['model'] == 'nvidia_lrelu_dropout_deep':
        return nvidia_lrelu_dropout_deep_model(model, args)       
    elif args['model'] == 'nvidia_lrelu_dropout_2':
        return nvidia_lrelu_dropout_2_model(model, args)            
    elif args['model'] == 'dev':
        return dev_model(model, args)


def nvidia_model(model, args):
    '''Model as defined in in the Nvida paper'''
    print("Using Nvidia Model.")

    model.add(Conv2D(3, kernel_size=(5, 5), strides=(2,2), activation=None))
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2,2), activation=None))
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2,2), activation=None))
    model.add(Conv2D(48, kernel_size=(3, 3), activation=None))
    model.add(Conv2D(64, kernel_size=(3, 3), activation=None))
    model.add(Flatten())
    model.add(Dense(100, activation=None))
    model.add(Dense(50, activation=None))
    model.add(Dense(10, activation=None))
    model.add(Dense(1)) # Output should be in range [-1, 1]

    # Minimize with MSE as a basic start for the regression model
    model.compile(optimizer='adam', loss=losses.mean_squared_error,
            metrics=['accuracy'])
    return model


def nvidia_conv_relu_model(model, args):
    '''Model as defined in in the Nvida paper, 
    with ReLU activations on the Convolutional layers
    '''
    print("Using nvidia_conv_relu Model.")

    model.add(Conv2D(3, kernel_size=(5, 5), strides=(2,2), activation="relu"))
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2,2), activation="relu"))
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2,2), activation="relu"))
    model.add(Conv2D(48, kernel_size=(3, 3), activation="relu"))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(Flatten())
    model.add(Dense(100, activation=None))
    model.add(Dense(50, activation=None))
    model.add(Dense(10, activation=None))
    model.add(Dense(1)) # Output should be in range [-1, 1]

    # Minimize with MSE as a basic start for the regression model
    model.compile(optimizer='adam', loss=losses.mean_squared_error,
            metrics=['accuracy'])
    return model


def nvidia_relu_model(model, args):
    '''Model as defined in in the Nvida paper, 
    with ReLU activations on the all layers
    '''
    print("Using nvidia_relu Model.")

    model.add(Conv2D(3, kernel_size=(5, 5), strides=(2,2), activation="relu"))
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2,2), activation="relu"))
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2,2), activation="relu"))
    model.add(Conv2D(48, kernel_size=(3, 3), activation="relu"))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(Flatten())
    model.add(Dense(100, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1)) # Output should be in range [-1, 1]

    # Minimize with MSE as a basic start for the regression model
    model.compile(optimizer='adam', loss=losses.mean_squared_error,
            metrics=['accuracy'])
    return model

def nvidia_conv_dropout_model(model, args):
    '''Model as defined in in the Nvida paper, 
    with ReLU activations on the all layers
    '''
    print("Using nvidia_conv_dropout_model Model.")

    model.add(Conv2D(3, kernel_size=(5, 5), strides=(2,2), activation="relu"))
    model.add(Dropout(0.8))
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2,2), activation="relu"))
    model.add(Dropout(0.8))
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2,2), activation="relu"))
    model.add(Dropout(0.8))
    model.add(Conv2D(48, kernel_size=(3, 3), activation="relu"))
    model.add(Dropout(0.8))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(Flatten())
    model.add(Dense(100, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1)) # Output should be in range [-1, 1]

    # Minimize with MSE as a basic start for the regression model
    model.compile(optimizer='adam', loss=losses.mean_squared_error,
            metrics=['accuracy'])
    return model

def nvidia_lrelu_model(model, args):
    '''Model as defined in in the Nvida paper, 
    with ReLU activations on the all layers. 

    Note to properly save/load LeakyReLU must be added as a seperate layer.
    '''
    print("Using nvidia_lrelu Model.")
    lrelu = LeakyReLU(alpha=0.3)

    model.add(Conv2D(3, kernel_size=(5, 5), strides=(2,2), activation=None))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2,2), activation=None))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2,2), activation=None))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Conv2D(48, kernel_size=(3, 3), activation=None))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Conv2D(64, kernel_size=(3, 3), activation=None))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Flatten())
    model.add(Dense(100, activation=None))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dense(50, activation=None))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dense(10, activation=None))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dense(1)) # Output should be in range [-1, 1]

    # Minimize with MSE as a basic start for the regression model
    model.compile(optimizer='adam', loss=losses.mean_squared_error,
            metrics=['accuracy'])
    return model


def simple_conv_1(model, args):
    '''Simple Convolutional model'''
    # Define Nvidia model
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(Flatten())
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1)) # Output should be in range [-1, 1]

    # Minimize with MSE as a basic start for the regression model
    model.compile(optimizer='adam', loss=losses.mean_squared_error,
            metrics=['accuracy'])
    return model


def nvidia_relu_nadam_model(model, args):
    '''Model as defined in in the Nvida paper, 
    with ReLU activations on the all layers and nadam optimizer
    '''
    print("Using nvidia_relu_nadam_model Model.")

    model.add(Conv2D(3, kernel_size=(5, 5), strides=(2,2), activation="relu"))
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2,2), activation="relu"))
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2,2), activation="relu"))
    model.add(Conv2D(48, kernel_size=(3, 3), activation="relu"))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(Flatten())
    model.add(Dense(100, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1)) # Output should be in range [-1, 1]

    # Minimize with MSE as a basic start for the regression model
    model.compile(optimizer='nadam', loss=losses.mean_squared_error,
            metrics=['accuracy'])
    return model


def inception_v3_model(model, args):
    '''Inception V3 model'''
    print("Using Pre-trained Inception V3 Model.")
        
    # Define Inception V3 model
    model.add(InceptionV3(weights="imagenet", include_top=False))
    model.add(Flatten())
    model.add(Dense(1)) # Output should be in range [-1, 1]

    # Minimize with MSE as a basic start for the regression model
    model.compile(optimizer='adam', loss=losses.mean_squared_error,
            metrics=['accuracy'])
    return model


def nvidia_lrelu_dropout_model(model, args):
    '''Model as defined in in the Nvida paper, 
    with ReLU activations on the all layers. 

    With dropout nodes after dense layers.
    Note to properly save/load LeakyReLU must be added as a seperate layer.
    '''
    print("Using nvidia_lrelu_dropout_model Model.")
    lrelu = LeakyReLU(alpha=0.3)

    model.add(Conv2D(3, kernel_size=(5, 5), strides=(2,2), activation=None))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2,2), activation=None))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2,2), activation=None))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Conv2D(48, kernel_size=(3, 3), activation=None))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Conv2D(64, kernel_size=(3, 3), activation=None))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation=None))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation=None))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation=None))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.5))
    model.add(Dense(1)) # Output should be in range [-1, 1]

    # Minimize with MSE as a basic start for the regression model
    model.compile(optimizer='adam', loss=losses.mean_squared_error,
            metrics=['accuracy'])
    return model

def nvidia_lrelu_dropout_2_model(model, args):
    '''Model as defined in in the Nvida paper, 
    with ReLU activationson the all layers. 

    With dropout nodes on all layers.

    Note to properly save/load LeakyReLU must be added as a seperate layer.
    '''
    print("Using nvidia_lrelu_dropout_2_model Model.")
    lrelu = LeakyReLU(alpha=0.3)

    model.add(Conv2D(3, kernel_size=(5, 5), strides=(2,2), activation=None))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.8))
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2,2), activation=None))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.8))
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2,2), activation=None))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.8))
    model.add(Conv2D(48, kernel_size=(3, 3), activation=None))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.8))
    model.add(Conv2D(64, kernel_size=(3, 3), activation=None))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.8))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation=None))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation=None))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation=None))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.5))
    model.add(Dense(1)) # Output should be in range [-1, 1]

    # Minimize with MSE as a basic start for the regression model
    model.compile(optimizer='adam', loss=losses.mean_squared_error,
            metrics=['accuracy'])
    return model


def nvidia_lrelu_dropout_deep_model(model, args):
    '''Model as defined in in the Nvida paper, with extra conv layers, 
    with ReLU activations on the all layers.
    Dropout layers added after Dense layers.  

    Note to properly save/load LeakyReLU must be added as a seperate layer.
    '''
    print("Using nvidia_lrelu_dropout_deep_model Model.")
    lrelu = LeakyReLU(alpha=0.3)

    model.add(Conv2D(3, kernel_size=(5, 5), strides=(2,2), activation=None))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2,2), activation=None))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2,2), activation=None))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Conv2D(48, kernel_size=(3, 3), activation=None))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Conv2D(64, kernel_size=(3, 3), activation=None))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Conv2D(96, kernel_size=(3, 3), activation=None))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Conv2D(128, kernel_size=(3, 3), activation=None))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation=None))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation=None))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation=None))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.5))
    model.add(Dense(1)) # Output should be in range [-1, 1]

    # Minimize with MSE as a basic start for the regression model
    model.compile(optimizer='adam', loss=losses.mean_squared_error,
            metrics=['accuracy'])
    return model


def dev_model(model, args):
    '''Dev Model being toyed with, not for real training.'''
    print("Using Dev Model.")

    lrelu = LeakyReLU(alpha=0.3)

    model.add(Conv2D(3, kernel_size=(5, 5), strides=(2,2), activation=None))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2,2), activation=None))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2,2), activation=None))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Conv2D(48, kernel_size=(3, 3), activation=None))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Conv2D(64, kernel_size=(3, 3), activation=None))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1)) # Output should be in range [-1, 1]

    # Minimize with MSE as a basic start for the regression model
    model.compile(optimizer='adam', loss=losses.mean_squared_error,
            metrics=['accuracy'])
    return model