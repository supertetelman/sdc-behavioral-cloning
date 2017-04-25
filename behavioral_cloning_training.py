import numpy as np
np.random.seed(9632) # We need to seed Numpy before importing Keras for reproducibility.
import tensorflow as tf
from keras.models import Sequential

import behavioral_cloning as bc
import behavioral_cloning_models as bc_models


def train_generic_model(args):
    '''Train and save a model'''
    # Create a Sequential Model
    model = Sequential()

    # Use pre-processing from library: color conversion, normalization, resizing, cropping
    model = bc.add_model_preprocess(model, args)

    # Get the model specified in args
    model = bc_models.get_model(model, args)

    # Use the library to train, save, plot,
    return bc.train_save_eval_model(model, args)


def main(_):
    '''Set args map to tf FLAGS and call model, used for running from cmd line'''
    # Get the default
    args = bc.default_args()

    # Update the default with model specific params
    args['output'] = flags.FLAGS.output
    args['epochs'] = flags.FLAGS.epochs
    args['validation'] = flags.FLAGS.validation
    args['test'] = flags.FLAGS.test
    args['correction'] = flags.FLAGS.correction
    args['all_images'] = flags.FLAGS.all_images
    args['nonormalize'] = flags.FLAGS.nonormalize
    args['nohorizontal'] = flags.FLAGS.nohorizontal
    args['novertical'] = flags.FLAGS.novertical
    args['debug'] = flags.FLAGS.debug
    args['save_history'] = flags.FLAGS.save_history
    args['show_history'] = flags.FLAGS.show_history
    args['crop'] = flags.FLAGS.crop
    args['eval_test'] = flags.FLAGS.eval_test
    args['eval_validation'] = flags.FLAGS.eval_validation
    args['batch'] = flags.FLAGS.batch
    args['yuv'] = flags.FLAGS.yuv
    args['blur'] = flags.FLAGS.blur
    args['resize'] = flags.FLAGS.resize

    # parse the args
    args = bc.parse_args(args)

    # Run the model
    return train_generic_model(args)


if __name__ == '__main__':
    # command line flags
    flags = tf.app.flags
    flags.DEFINE_string('output', "nvidia_model.h5", "The file name to save the trained model to.")
    flags.DEFINE_string('model', "nvidia", "The name of the model to train on.")
    flags.DEFINE_integer('epochs', 10, "The number of epochs.")
    flags.DEFINE_float('validation', 0.2, "Validation split size.")
    flags.DEFINE_float('test', 0.1, "Test split size.")
    flags.DEFINE_float('correction', False, "The Correction to use on the left/right images.")
    flags.DEFINE_bool('all_images', False, "Use the Left, Right, and Center images")
    flags.DEFINE_bool('nonormalize', False, "Do not nomarlize the images")
    flags.DEFINE_bool('nohorizontal', False, "Do not add horizontally flipped copies of all the images")
    flags.DEFINE_bool('novertical', False, "Do not add vertically flipped copies of all the images")
    flags.DEFINE_bool('debug', False, "Add debug prints and show images at various steps")
    flags.DEFINE_bool('save_history', False, "Save the History for the run.")
    flags.DEFINE_bool('show_history', False, "Show the History plot for the run.")
    flags.DEFINE_bool('crop', False, "Crop the images before processing.")
    flags.DEFINE_bool('eval_test', False, "Evaluate the test data.")
    flags.DEFINE_bool('eval_validation', False, "Evaluate the Validation data.")
    flags.DEFINE_bool('yuv', False, "Convert RGB to YUV.")
    flags.DEFINE_bool('gray', False, "Convert RGB to gray.")
    flags.DEFINE_bool('blur', False, "Blur Image")
    flags.DEFINE_bool('resize', False, "Resize Image")
    flags.DEFINE_integer('batch', 512, "The batch size.")
    tf.app.run()   
