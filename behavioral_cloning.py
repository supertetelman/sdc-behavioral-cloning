'''library to parse formatted training data csv, augment data, etc.'''
import csv
import os
import cv2
import numpy as np

import matplotlib.pyplot as plt
import pickle
import time
from PIL import Image
import matplotlib.image as mpimg

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import keras
from keras.layers import  Cropping2D, Lambda
import tensorflow as tf


############################################################################
##########    Global directory paths and meta-data    ######################
############################################################################
DATA_DIR = 'data'
LOG_DIR = "logs"
MODEL_DIR = "models"
CSV_DIR = os.path.join(DATA_DIR, "csv")
SAMPLE_IMAGE = os.path.join(DATA_DIR, "test_image.jpg")
ANGLE_INDEX = 3


############################################################################
####################    Helper Functions    ################################
############################################################################
def _get_image_path(data_dir, source_path):
    '''Images are kept in format ./data/<run>/IMG/<filename>, this function parses that.
    It is necessary to return a relative filename path because these images have been moved
    since they were recorded and the absolute path in the CSV files is no longer valid.
    '''
    relative_filename = os.sep.join(source_path.split(os.sep)[-3:]) # Should be "<run>/IMG/<filename>""
    return os.path.join(data_dir, relative_filename)

def _make_image_path(data_dir, source_path, new_run):
    '''Images are kept in format ./data/<run>/IMG/<filename>, this function generates that.
    Used when creating augmented data.
    '''
    relative_filename = os.sep.join([new_run] + source_path.split(os.sep)[-2:]) # Should be "<new_run>/IMG/<filename>""
    return os.path.join(data_dir, relative_filename)


def _get_corrections(all_images, corrections):
    '''Convert False/float correction to  3 element list
    all_images False -> No corrections
    corrections None -> Default, False -> None, Float -> [0, c, c * -1], List -> noop
    '''
    if all_images:
        indexes = [0, 1, 2]

        # If corrections is False, don't use them, if None default them, if True keep them
        if corrections is None:
            corrections = [0.0, 0.1, -0.1]
        elif corrections is False:
            corrections = [0.0, 0.0, 0.0]
        elif isinstance(corrections, float):
            corrections = [0, corrections, -corrections]
    else:
        indexes = [0]
        corrections = [0]
    assert len(indexes) == len(corrections), " \
            corrections and index image lists must be same length. %s:%s" %(corrections, indexes)
    return corrections, indexes


############################################################################
##################    Data Reading Functions    ############################
############################################################################
def read_data_lines(data_csv):
    '''Given a csv file or list of csv file return a list containing all lines'''
    lines = [] # Store each full training line 
   
    # If only 1 csv was specified make it a list
    if type(data_csv) is str:
        data_csv = [data_csv]

    # Iterate over each csv file
    for current_data_csv in data_csv:
        # Read each line of the csv file
        with open(current_data_csv) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lines.append(line)

    return lines


def split_data_lines(lines, test=0.1, validation=0.2):
    '''Shuffle and Split sample lines into train/validation/test lists'''
    print("Splitting lines (100%%, %d) into: ..." %(len(lines)))
    shuffle(lines)
    train_features, tmp_features = train_test_split(lines,
            test_size = (test + validation))
    validation_features, test_features = train_test_split(tmp_features,
            test_size = (test/(validation + test)))
    print("train dataset (%0.2f %%, %d), \
        \nvalidation dataset (%0.2f %%, %d), and \
        \ntest dataset (%0.2f %%, %d)" %(100 - (100 * (validation + test)),
            len(train_features), 100*validation,
            len(validation_features), 100*test, len(test_features)))
    return train_features, validation_features, test_features


def read_data(data_csv, all_images=True, corrections=None):
    '''Given a training csv file return parsed numpy arrays
    :param data_csv: csv filename or list of filenames formatted with each line as
        image_l, image_c, image_r, angle, throttle, break, speed
    :param all_images: True to use l, r, and c image. False to only use center image.

    :outputs X_train, y_train, lines:
    an X_train numpy array containing an image per line
    an y_train numpy array containing a steering angle per line
    a line list containing each line in the csv
    '''
    # Create lists 
    lines = [] # Store each full training line 
    images = [] # Store each training image
    measurements = [] # Store each training angle measurement
    
    # Handle usage of left/right/center images and corrections
    corrections, indexes = _get_corrections(all_images, corrections)

    # If only 1 csv was specified make it a list
    if type(data_csv) is str:
        data_csv = [data_csv]

    # Iterate over each csv file
    for current_data_csv in data_csv:
        print("Reading values from %s" %(current_data_csv))
        # Read each line of the csv file
        with open(current_data_csv) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lines.append(line)
                # Iterate over the left, center, and right images
                for index, correction in zip(indexes, corrections): 
                    image_path = _get_image_path(DATA_DIR, line[index])
                    image = cv2.imread(image_path)
                    images.append(image)
                    measurements.append(float(line[ANGLE_INDEX]) + correction) # XXX: This will dupe data for each c/l/r pair
    return np.asarray(images), np.asarray(measurements), lines


def read_data_generator(lines, batch_size=32, all_images=True, corrections=None, preprocess={}):
    '''Given a training csv file return parsed numpy arrays
    :param lines: output from read_data_lines()
    :param all_images: True to use l, r, and c image. False to only use center image.
    :param batch_size: size of each batch
    :param preprocess: a map containing preprocessin parameters, see default_preprocess()
    :outputs (X_train, y_train) of len() batch_size
    an X_train numpy array containing an image per line
    an y_train numpy array containing a steering angle per line
    '''
    # Count for batches
    count = 0
    total_count = 0

    # shuffle all the training lines
    shuffle(lines)

    # Loop forever for Generator    
    while 1:
        # Handle usage of left/right/center images and corrections
        corrections, indexes = _get_corrections(all_images, corrections)

        # Create lists 
        images = [] # Store each training image
        measurements = [] # Store each training angle measurement
        for line in lines:
            if line == []:
                continue # Ignore blank lines
            # Iterate over the left, center, and right images
            for index, correction in zip(indexes, corrections): 
                image_path = _get_image_path(DATA_DIR, line[index])
                image = cv2.imread(image_path)
                assert image is not None, "An image failed to load properly: %s" %(image_path)
                image = image_preprocess(image, preprocess)
                images.append(image)
                measurements.append(float(line[ANGLE_INDEX]) + correction)
                count += 1 # Increment count
                total_count += 1
                if count == batch_size:
                    # Yield the current batch and reset the lists/counts
                    count = 0

                    yield shuffle(np.asarray(images), np.asarray(measurements))
                    measurements = [] 
                    images = []

    # Yeild the remaining images
    print("\nTotal images processed by generator this EPOCH was: %d\n" %(total_count))
    yield shuffle(np.asarray(images), np.asarray(measurements))


############################################################################
##################    Data Augmentation Functions    ############################
############################################################################
def augment_data_flip_horiz(X, y):
    '''Return X, y with horizontally f
    lipped images from a 4D numpy array.
    :param X: numpy image array
    :param y: numpy training angle array
    :output X, y: Will be twice the original size with  flipped and non-flipped images
    '''
    # Return two numpy arrays with the original data stacked on-top of the horizontally flipped data
    return np.vstack((X, np.flip(X, axis=2))), np.hstack((y, -y))


def augment_data_flip_vert(X, y):
    '''Return X, y with verticaly flipped images from a 4D numpy array.
    :param X: numpy image array
    :param y: numpy training angle array
    :output X, y: Will be twice the original size with  flipped and non-flipped images
    '''
    # Return two numpy arrays with the original data stacked on-top of the vertically flipped data
    return np.vstack((X, np.flip(X, axis=1))), np.hstack((y, -y))


def augment_data_flip(X, y):
    '''Return X, y with verticaly and flipped images
    :param X: numpy image array
    :param y: numpy training angle array
    :output X, y: Will be 4x the original size with  flipped and non-flipped images
    '''
    return augment_data_flip_horiz(augment_data_flip_vert(X, y))


def augment_data_flip_horiz_alt(X, y):
    '''Return X, y with horizontally flipped images from a 4D numpy array.
    Alternate less efficient implementation using cv2
    :param X: numpy image array
    :param y: numpy training angle array
    :output X, y: Will be twice the original size with  flipped and non-flipped images
    '''
    images = []
    measurements = []
    for image, measurement in zip(X, y):
        # Add original images
        images.append(image) # XXX: This will duplicate data in-memory while X and images both exist
        measurements.append(measurement)

        # Add horizontally flipped image/measurement 
        images.append(cv2.flip(image, 1))
        measurements.append(measurement * -1.0)
    return np.asarry(images), np.asarray(measurements)


def augment_data_flip_vert_alt(X, y):
    '''Return X, y with verticaly flipped images from a 4D numpy array.
    Alternate less efficient implementation using cv2
    :param X: numpy image array
    :param y: numpy training angle array
    :output X, y: Will be twice the original size with  flipped and non-flipped images
    '''
    images = []
    measurements = []
    for image, measurement in zip(X, y):
        # Add original images
        images.append(image) # XXX: This will duplicate data in-memory while X and images both exist
        measurements.append(measurement)

        # Add horizontally flipped image/measurement 
        images.append(cv2.flip(image, 0))
        measurements.append(measurement * -1.0)
    return np.asarry(images), np.asarray(measurements)


def augment_data_flip_horiz_by_file(csv_files):
    '''Given a list of CSV files this will iterate through each line and create a _flip dataset
    This is done out of band of training and modeling and should be run once per dataset.
    The result wil be a ./csv/<run_flip>.csv file and a ./<run>_flip data dir.
    The new data will be horizontally flipped with corrected measurements
    '''
    # Fix the case of a single file being entered as a non-list
    if not isinstance(csv_files, list):
        csv_files = [csv_files]

    # Iterate over the list of files
    for csv_file in csv_files:
        # Check if _flip already in name
        if "_flip" in csv_file:
            continue

        # Remove .csv from the csv_file name to get the run_name
        old_run_name = csv_file[:-4] 
        old_file = os.path.join(CSV_DIR, csv_file)
        run_name = old_run_name + "_flip"
        new_file = os.path.join(CSV_DIR, run_name + ".csv")

        # Don't reflip data
        if os.path.isfile(new_file):
            continue

        print("Opening %s from run %s and creating %s for run %s" %(old_file, old_run_name, new_file, run_name))
        # Open the original CSV
        with open(old_file, 'r') as original_csv:
            # Create new csv file with _flip in name
            flipped_csv = open(new_file, 'w')

            # Create new image directory with _flip in name
            os.makedirs(os.path.join(DATA_DIR, run_name, "IMG"))

            # Create CSV reader/writer
            reader = csv.reader(original_csv)
            writer = csv.writer(flipped_csv)

            # Iterate over CSV file
            for line in reader:
                new_line = []
                # Iterate over C/L/R image
                for idx in range(0,3):
                    img = _get_image_path(DATA_DIR, line[idx])
                    new_img = _make_image_path(DATA_DIR, img, run_name) 
                    img = Image.open(img) # Load Image
                    img = img.transpose(Image.FLIP_LEFT_RIGHT) # Flip Image
                    img.save(new_img) # Save Image

                    # Add the image name to the csv
                    new_line.append(new_img)

                # Add a flipped measurement, and the rest of the values
                new_line.append(-float(line[ANGLE_INDEX]))
                new_line += line[4:]

                # Write the new line to the new file
                writer.writerow(new_line)
            flipped_csv.close()


############################################################################
#########    History Reading, Writing, and Plotting    #####################
############################################################################
def save_log(obj, name, title="logs"):
    '''Save a file to the logdir'''
    log_file = os.path.join(LOG_DIR, name)
    print("Saving %s to %s." %(title, log_file))
    pickle.dump(obj, open(log_file, 'wb'))


def save_history(history_object, name, timestamp=True, saveplot=False):
    '''Generate a timestamped name and serialize history object to file
    File saved in log_dir to <name>-history-<time>.p or to name if timestampe=False
    saveplot will save a plot with the same file name and a png extension
    '''
    # Set the name off a timestamp
    if timestamp:
        name = "%s-history-%d.p" %(name, int(time.time()))

    # Save to disk
    save_log(history_object.history, name, "history data")

    # Save a plot with a similar name
    if saveplot:
        show_history(history_object.history, name + ".png")


def import_history(history_file):
    '''Load a history from file within log_dir'''
    history_file = os.path.join(LOG_DIR, history_file)
    print("Loading history data to %s" %(history_file))
    return pickle.load(open(history_file, 'rb'))


def show_history(history, save=False, close_plot=True):
    '''plot the training and validation loss for each epoch 
    Optionally save it to a file instead of showing it.
    Optionally keep the plot open to overlay the next plot on the same figure.
    '''
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.ylim((0,0.08)) # XXX: Experimentally decent high end for loss

    if save:
        save = os.path.join(LOG_DIR, save)
        print("Saving history data to %s" %(save))
        plt.savefig(save)
    else:
        if close_plot:
            print("Displaying history data")
            plt.show()
    if close_plot:
        plt.close()


def model_save(model, output):
    # XXX: May fail on Windows if Keras is not patched with this fix:
    # BUG: https://github.com/fchollet/keras/issues/4135
    output_file = os.path.join(MODEL_DIR, output)
    print("Saving model to %s" %(output_file))

    model.save(output_file)


############################################################################
######################    Image Processing  ################################
############################################################################
def image_preprocess(image, preprocess):
    '''Image pre-processing that is not yet implemented in the model.
    Ideally this will all be moved into the model.
    '''
    if preprocess.get('yuv', False):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    if preprocess.get('blur', False):
        image = cv2.GaussianBlur(image, preprocess['blur'], 0)
    return image


def normalization(img_shape):
    return Lambda(lambda x: (x / 255.0) - 0.5, input_shape=img_shape)


def cropping(crop_size):
    return Cropping2D(cropping=(crop_size, (0,0)))


def resize_helper(x, **arguments):
    # XXX: Lambda Layers/Functions with external library calls must be defined in a helper function
    # BUG: https://github.com/fchollet/keras/issues/4609 
    import tensorflow as tf
    return tf.image.resize_images(x, arguments['resize_shape'])


def resize(resize_shape):
    return Lambda(resize_helper, arguments={'resize_shape' : resize_shape})


def grayscale_helper(x, **arguments):
    # XXX: Lambda Layers/Functions with external library calls must be defined in a helper function
    # BUG: https://github.com/fchollet/keras/issues/4609 
    import tensorflow as tf
    return tf.image.rgb_to_grayscale(x)


def grayscale():
    return Lambda(grayscale_helper, arguments={})


def add_model_preprocess(model, args):   
    # Normalize the images
    if not args['nonormalize']:
        model.add(normalization(args['img_shape']))
        
    # Crop the images
    if args['crop']:
        model.add(cropping(args['crop_size'])) 
    
    # Resize the image after cropping it
    if args['resize']:
        model.add(resize(args['resize_shape']))

    if args['gray']:
        model.add(grayscale())

    return model

############################################################################
#####################   Model Training and Evalutation  ####################
############################################################################
def train_save_eval_model(model, args):
    '''Given a model and args, create data generators, 
    run training, generate some plots, save the model, and evalutate against test/validation
    '''
    # Setup Generators and data lists
    lines = read_data_lines(args['data_csv_list'])
    train_samples, validation_samples, test_samples = split_data_lines(lines, 
            args['test'],  args['validation'])
    train_generator = read_data_generator(train_samples, args['batch'], 
            args['all_images'], args['correction'], args['preprocess'])
    validation_generator = read_data_generator(validation_samples, args['batch'], 
            args['all_images'], args['correction'], args['preprocess'])
    predict_generator = read_data_generator(test_samples, args['batch'], 
            args['correction'], args['preprocess'])

    # Train & save the model.
    print("\nTraining %s for %d epochs with \n %.1f %% data saved for validation.\
            \n\tand \n %.1f %% data saved for test \n\tand \na batch size of %d \
            " %(args['output'],  args['epochs'], args['validation'] * 100, 
            args['test'] * 100, args['batch']))

    history_object = model.fit_generator(train_generator, 
            steps_per_epoch = len(train_samples) / args['batch'], 
            epochs = args['epochs'], validation_data = validation_generator, 
            validation_steps = len(validation_samples) / args['batch'],
            )

    # Save the history and history plot to timestamped files
    if args['save_history']:
        save_history(history_object, args['output'] + ".p", True, True)
    
    # Display the history plot
    if args['show_history']:
        show_history(history_object.history, False)
    
    # Save the model
    model_save(model, args['output'])

    # Run the final test evalutation
    if args['eval_test']:
        print("Evaluating test.")
        test_loss, test_acc = model.evaluate_generator(predict_generator, 
                steps = len(test_samples) / args['batch'])
        print("Test Loss: %0.4f, Test Accuracy: %0.4f" %(test_loss, test_acc))

    # Run and return the validation loss/acc
    val_loss, val_acc = -1, -1
    if args['eval_validation']:
        print("Evaluating Validation.")
        val_loss, val_acc = model.evaluate_generator(validation_generator, 
                steps = len(validation_samples) / args['batch'])
        print("Validation Loss: %0.4f, Validation Accuracy: %0.4f" %(val_loss, val_acc))
        
    # XXX: Depending on some threading issues saving could cause issues unless 
    # We clear backend (TF) sessions when done using the model.
    # BUG: https://github.com/tensorflow/tensorflow/issues/3388
    keras.backend.clear_session() 

    return val_loss, val_acc


############################################################################
########   Command Line arguments, defaults, parsing, and logging  #########
############################################################################
def default_preprocess():
    preprocess = {}
    preprocess['yuv'] = False
    preprocess['blur'] = False
    return preprocess


def default_args():
    '''Generate default args map, used for running  models from external python calls'''
    args = {}
    args['output'] = "model.h5"
    args['epochs'] = 10
    args['validation'] = 0.2
    args['test'] = 0.1
    args['correction'] = False
    args['all_images'] = False
    args['nonormalize'] = False
    args['nohorizontal'] = False
    args['novertical'] = False
    args['debug'] = False
    args['save_history'] = False 
    args['show_history'] = False
    args['crop'] = False
    args['eval_test'] = False
    args['eval_validation'] = True
    args['batch'] = 32
    args['blur'] = False
    args['yuv'] = False
    args['gray'] = False

    # Hardcoded image shape, crop values, resize values, etc.
    args['img_shape'] = (160, 320, 3) 
    args['crop_size'] = (70,25) # TODO: Tune this
    args['resize_shape'] = (200, 66) 
    args['blur_kernel'] = (3, 3) # TODO: Tune this
    args['resize'] = False

    # By default use all files in ./<data_dir>/csv/*
    args['csv_file_names'] = os.listdir(CSV_DIR) # Files contained in ./data/csv/*.csv
    args['data_csv_list'] = []

    args['preprocess'] = default_preprocess()
    return args


def parse_args(args):
    '''Print some helper text based on args, and update corrections'''  
    # Convert the csv file names into valid relative-path files
    data_csv_list = []
    for csv_file in args['csv_file_names']:
        data_csv_list.append(os.path.join(CSV_DIR, csv_file))
    args['data_csv_list'] = data_csv_list
    print("CSV files used for training will be %s" %(str(args['data_csv_list'])))

    if args['all_images']:
        print("Using left, right, and center images.")
    else:
        print("Only using center images.")

    # Setup the correction for the right/left image
    if args['correction']:
        args['correction'] = [0, args['correction'], args['correction'] * -1] # TODO: Allow left/right correction to differ?
        print("Using these correction values: \n%s" %(str(args['correction'])))
    else:
        print("Not using any correction values.")
        args['correction'] = False

    if not args['nonormalize']:
        print("Normalizing Images before processing.")

    if args['crop']:
        print("Cropping Images before processing: %s" %(str(args['crop_size'])))

    if args['yuv']:
        print("Converting RGB Images to YUV")
        args['preprocess']['yuv']  = True

    if args['gray']:
        print("Converting RGB Images to grayscale")

    if args['resize']:
        print("Resizing images to %s" %(str(args['resize_shape'])))

    if args['blur']:
        print("Blurring Images with Kernel %s" %(str(args['blur_kernel'])))
        args['preprocess']['blur']  = args['blur_kernel']

    if args['nohorizontal'] is False:
        print("Verifying that a flipped dataset exists, creating one if it doesn't")
        augment_data_flip_horiz_by_file(args['csv_file_names'])

    return args
