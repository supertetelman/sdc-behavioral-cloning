import copy
import pickle
import os
import glob
import time

import behavioral_cloning as bc
import behavioral_cloning_training as bc_train


##### Display, Plotting, and Results viewing helper functions #####
def display_sorted_results(results_files):
    '''Take a list of results file names and return a sorted list 
    containing loss scores and args for each run.
    '''
    all_results = [] # A list of results from all files

    # Iterate through each results file and add them to the list
    for results_file in results_files:
        results = pickle.load(open(results_file, 'rb'))
        print("Parsing %s with %d entries" %(results_file, len(results)))
        for result in results:
            # IF it is a bad entry, don't fail. Just throw it out
            try:
                assert result is not None, "Error loading" # TODO: cleanup duplicated asserts
                assert len(result) == 4, "Improper size fore result"
                assert 'model' in result[3], "Model name missing"
                assert isinstance(result[2], int), "Count expected but not found"
            except:
                continue
            all_results.append(result)

    # Sort all of the results by Loss  and return the sorted list
    all_results.sort(key=lambda tup: tup[0])
    return all_results


def average_by_count(results):
    '''Given a list of results, return an aggregate map of average_loss by run count id'''
    assert len(results) > 0

    # Final results go here
    result_average = {}

    # Sort results by key (count)
    results.sort(key=lambda tup: tup[2]) 

    # Iterate through the sorted list and average the Loss across all runs with identical key
    key = -1 # Initialize key to bad value
    for result in results:
        if result[2] != key: # If the key has changed, reset
            if key >= 0:
                result_average[key] = loss_sum / count # Average
            loss_sum = 0
            count = 0  
            key = result[2] # Update key
        count += 1
        loss_sum += result[0]
    result_average[key] = loss_sum / count # Average the last key
  
    # TODO: result_average = [(id, result_average[id]) for id in sorted(result_average, key=result_average.get)])
    return result_average


def plot_run_files(results_pickle_file, history_dir, close_plot=False):
    '''Given a results pickle file and a directory of history files show plots
    If results_pickle_file is False just try to import all .p files in the directory

    Setting close_plot == False will put all plots in the same figure.
    '''
    # In case the run did not complete correctly we plot with best effort
    if results_pickle_file == False:
        print("No results file specified, searching for all *.p files in %s." %(history_dir))
        history_pickle_files = glob.glob(os.path.join(history_dir, '*.p'))
        for history_file in history_pickle_files:
            # Import and display the history
            if close_plot:
                print("Run: %s" %(history_file))
            history = pickle.load(open(history_file, 'rb'))
            try:
                bc.show_history(history, save=False, close_plot=close_plot)
            except:
                print("Had an issue plotting %s" %(history_file))
        return 2

    # Load the pickled results, these are sorted in order by lowest loss
    results = pickle.load(open(results_pickle_file, 'rb'))

    # Iterate through all the results
    for result in results:
        assert result is not None, "Error loading"
        assert len(result) == 4, "Improper size fore result"
        assert 'model' in result[3], "Model name missing"
        assert isinstance(result[2], int), "Count expected but not found"

        # Example: nvidia-1, nvidia-lrelu-1
        model_id = "%s-%d*.p" %(result[3]['model'], result[2])

        # Search for all files matching the id
        result_pickle_file = glob.glob(os.path.join(history_dir, model_id))

        # There should only be one ... We take the first in case of extras
        if len(result_pickle_file) > 1:
            print("Found multiple result files for %s, taking the first" %(result[3]['output']))
        result_pickle_file = result_pickle_file[0]

        # Import and display the history
        if close_plot:
            print("Run: %d" %(result[2]))
        history = pickle.load(open(result_pickle_file, 'rb'))
        bc.show_history(history, save=False, close_plot=close_plot)


##### Tuning Loops #####
def _single_run(args, previous_args):
    '''Function to take args and run train the model with some basic error checking'''
    try:
        # Parse the args
        args = bc.parse_args(args)

        # Print a diff of the args for easier debugging
        if previous_args is not None:
            for key in previous_args.keys():
                if previous_args.get(key, -1) != args.get(key, -2):
                    print("This run has changed the value of %s from %s to %s" %(key,
                            str(previous_args[key]), str(args[key])))

        # Copy the args so we can save them cleanly
        current_args = copy.copy(args)

        # parse the args
        loss, acc = bc_train.train_generic_model(current_args)
        print('----------------------')
        print("Accuracy of **%0.4f** \
                \nLoss of **%0.4f** found for \
                \n %s" %(acc, loss, str(current_args)))
    except Exception as err:
        for i in range(10):
            print('----------------------')
        print(err) # Probably an OOM or an invalid options combo
        print('----------------------')
        print(current_args) 
        for i in range(10):
            print('----------------------')
        loss, acc = -1, -1
    return loss, acc, current_args


def _hypertune_loop(results_file, model_name, show_history, count_list, csv_file_names, options):
    '''Big loop to train models against a combination of parameters'''
    start = time.time()

    # Get the default args
    args = bc.default_args()

    # Set some static/inital values
    results = []
    args['test'] = 0.001 # Don't save any test data
    args['save_history'] = True # Save the history and plot an image
    args['model'] = model_name
    args['show_history'] = show_history
    if csv_file_names is None: # If None use these here
        args['csv_file_names'] = ['t1.csv', 't1-rev.csv']
    elif csv_file_names: # If False, use the default, else assign it.
        args['csv_file_names'] = csv_file_names


    count = 0
    previous_args = None

    # Loop through all the options
    for nohorizontal in options['nohorizontal']:
        args['nohorizontal'] = nohorizontal
        for novertical in options['novertical']:
            args['novertical'] = novertical
            for crop in options['crop']:
                args['crop'] = crop
                for yuv in options['yuv']:
                    args['yuv'] = yuv
                    for blur in options['blur']:
                        args['blur'] = blur    
                        for gray in options['gray']:
                            args['gray'] = gray
                            for all_images in options['all_images']:
                                args['all_images'] = all_images
                                for epochs in options['epochs']:
                                    args['epochs'] = epochs
                                    for batch in options['batch']:
                                        args['batch'] = batch
                                        for validation in options['validation']:
                                            args['validation'] = validation
                                            corrections = [False] # False is the only things that works without all_images
                                            if all_images:
                                                corrections = options ['corrections']
                                            for correction in corrections: # XXX: Previously after all_images
                                                args['correction'] = correction
                                                count += 1
                                                args['output'] = "%s-%d.h5" %(args['model'], count)

                                                # Specify specific iterations to run
                                                if count_list and count not in count_list:
                                                    continue

                                                # Run the model past/current args, and save off the results to a list
                                                loss, acc, previous_args = _single_run(args, previous_args)
                                                results.append((loss, acc, count, previous_args))

                                                # Print elapsed time
                                                print("Taken a total of %d seconds" %(int(time.time() - start)))
    # Sort and print the results
    results.sort(key=lambda tup: tup[0])
    for result in results:
        print(result)

    # Save the results
    bc.save_log(results, results_file)

    # Print elapsed time
    print("Finished after a total of %d seconds" %(int(time.time() - start)))

def hypertune_model(results_file, model_name, show_history = False, count_list = False, csv_file_names = None):
    options = {}
    options['nohorizontal'] = [False] # Was True
    options['novertical'] = [True]
    options['crop'] = [True, False]
    options['yuv'] = [False, True]
    options['blur'] = [False, True]
    options['gray'] = [False, True]
    options['all_images'] = [False, True]
    options['corrections'] = [False, .1, .25]
    options['epochs'] = [5] # Was 10
    options['batch'] = [128]
    options['validation'] = [0.2]
    return _hypertune_loop(results_file, model_name, show_history, count_list, csv_file_names, options)


def hypertune_model_2(results_file, model_name, show_history = False, count_list = False, csv_file_names = None):

    options = {}
    options['nohorizontal'] = [False]
    options['novertical'] = [True]
    options['crop'] = [False]
    options['yuv'] = [False]
    options['blur'] = [False]
    options['gray'] = [False]
    options['all_images'] = [True]
    options['corrections'] = [0.25]
    options['epochs'] = [20]
    options['batch'] = [32, 128, 512]
    options['validation'] = [0.2, 0.3, 0.5]
    return _hypertune_loop(results_file, model_name, show_history, count_list, csv_file_names, options)


def final_model(results_file, model_name, show_history = False, count_list = False, csv_file_names = None):
    options = {}
    options['nohorizontal'] = [False]
    options['novertical'] = [True]
    options['crop'] = [False]
    options['yuv'] = [False]
    options['blur'] = [False]
    options['gray'] = [False]
    options['all_images'] = [True]
    options['corrections'] = [0.25]
    options['epochs'] = [50]
    options['batch'] = [128]
    options['validation'] = [0.3]
    return _hypertune_loop(results_file, model_name, show_history, count_list, csv_file_names, options)


if __name__ == '__main__':
    # Some of these runs will override each others models. This is done to ensure that inspected
    # and saved off models are not accidentally removed and far too much disk space is not consumed.

    # hypertune_model("nvidia_model.p", "nvidia")
    # hypertune_model("nvidia_lrelu.p", "nvidia_lrelu", False)
    # hypertune_model("simple_conv_1.p", "simple_conv_1", False)
    # hypertune_model("nvidia_conv_relu.p", "nvidia_conv_relu", False)
    # hypertune_model("nvidia_relu.p", "nvidia_relu", False)
    # hypertune_model("nvidia_relu_nadam_model.p", "nvidia_relu_nadam_model", False)
    # hypertune_model("nvidia_conv_dropout_mouse.p", "nvidia_conv_dropout", False)



    # hypertune_model("nvidia_model_mouse.p", "nvidia", False, False, csv_file_names = ["mouse-t1.csv"])
    # hypertune_model("nvidia_lrelu_mouse.p", "nvidia_lrelu", False, False, csv_file_names = ["mouse-t1.csv"])
    # hypertune_model("simple_conv_1_mouse.p", "simple_conv_1", False, False, csv_file_names = ["mouse-t1.csv"])
    # hypertune_model("nvidia_conv_relu_mouse.p", "nvidia_conv_relu", False, False, csv_file_names = ["mouse-t1.csv"])
    # hypertune_model("nvidia_relu_mouse.p", "nvidia_relu", False, False, csv_file_names = ["mouse-t1.csv"])
    # hypertune_model("nvidia_relu_nadam_model_mouse.p", "nvidia_relu_nadam_model", False, False, csv_file_names = ["mouse-t1.csv"])
    # hypertune_model("nvidia_conv_dropout_mouse.p", "nvidia_conv_dropout", False, False, csv_file_names = ["mouse-t1.csv"])

    # hypertune_model("inception_v3_model_mouse.p", "inception_v3_model", False, False, csv_file_names = ["mouse-t1.csv"])
    
    # hypertune_model_2("nvidia_conv_dropout_tune.p", "nvidia_conv_dropout", False, False, False)
    # hypertune_model_2("nvidia_conv_dropout_tune.p", "nvidia_conv_dropout", False, False, False)
    
    # hypertune_model_2("nvidia_lrelu.p", "nvidia_lrelu")
    # final_model("nvidia_lrelu.p", "nvidia_lrelu")
    
    # final_model("dev.p", "dev")
    # final_model("nvidia_lrelu_dropout_2_model.p", "nvidia_lrelu_dropout_2")
    # final_model("nvidia_lrelu_dropout_model.p", "nvidia_lrelu_dropout")
    # final_model("nvidia_lrelu_dropout_model.p", "nvidia_lrelu_dropout", False, False, False)
    # final_model("dev.p", "dev", False, False, False)
    
    # hypertune_model("nvidia_lrelu_hyper.p", "nvidia_lrelu", False, False, False)
    # hypertune_model("nvidia_lrelu_dropout_deep.p", "nvidia_lrelu_dropout_deep_", False, False, False)
    # final_model("nvidia_lrelu_dropout_deep_final.p", "nvidia_lrelu_dropout_deep", False, False)
    # final_model("nvidia_lrelu_final.p", "nvidia_lrelu", False, False, False)
    pass
