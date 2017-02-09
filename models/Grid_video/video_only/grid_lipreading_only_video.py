from __future__ import print_function
import numpy as np
import glob
from collections import defaultdict
import theano.tensor as T
import lasagne
import theano
import time
import sys
import argparse
import os
from lipreading_utils import find_data_mean, find_data_std, HDF5VideoOnlyLipReadingIterator, ftensor5, get_train_and_val_fn, vocab_size, print_model
from sensors_ini.rnns.lasagne_utils import save_model, store_in_log, load_model, load_log, replace_updates_nans_with_zero
from lipreading_models import get_dan_video_only
import h5py

# Example:
#   ipython --pdb lipreading_training.py -- --do_norm 0 \
#                --aud_data_dir /home/dneil/datasets/grid/audio/np_data/ \
#                --vid_data_dir /home/dneil/datasets/grid/video/np_data/ \
#                --mean_filename mean_full --std_filename std_full \
#                --run_id full_more_aud_loss_cont --num_epochs 100 \
#                --resume lipreading_4.24_full_more_aud_loss_cont_recent \
#                --test_blankout 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a lipreading network.')
    # File and path naming stuff
    parser.add_argument('--h5file',       default='/home/dneil/rd/grid_40ms.hdf5', help='HDF5 File that has the data.')
    parser.add_argument('--run_id',       default=os.environ.get('LSB_JOBID','default'), help='ID of the run, used in saving.  Gets job ID on Euler, otherwise is "default".')
    parser.add_argument('--filename',     default='lipreading_29.8.16', help='Filename to save model and log to.')
    parser.add_argument('--resume',       default=None, help='Filename to load model and log from.')
    # Control meta parameters
    parser.add_argument('--seed',         default=42, type=int, help='Initialize the random seed of the run (for reproducibility).')
    parser.add_argument('--batch_size',   default=128, type=int, help='Batch size.')
    parser.add_argument('--num_epochs',   default=100, type=int, help='Number of epochs to train for.')
    parser.add_argument('--patience',     default=4, type=int, help='How long to wait for an increase in validation error before quitting.')
    parser.add_argument('--patience_key', default='val_acc', help='What key to look at before quitting.')
    parser.add_argument('--wait_period',  default=100, type=int, help='How long to wait before looking for early stopping.')
    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)

    # Set the save name
    comb_filename = args.filename + '_' + args.run_id

    # Load dataset
    dataset = h5py.File(args.h5file, "r")

    # Instantiate iterator
    d = HDF5VideoOnlyLipReadingIterator()

    # Create symbolic vars
    vid_in   = ftensor5('vid_in')
    vid_mask = T.bmatrix('vid_mask')
    targets  = T.fmatrix('targets')

    # Build model
    print("Building network ...")
    #   Get input dimensions
    network = get_dan_video_only(vid_in)
    # Instantiate log
    log = defaultdict(list)
    print("Built.")

    # Resume if desired
    if args.resume:
        print('RESUMING: {}'.format(args.resume))
        load_model(args.resume, network)
        log = load_log(args.resume)

    # Dump some debug data if we like
    print_model(network)

    # Compile the learning functions
    print('Compiling functions...')
    train_fn, val_fn, out_fn = get_train_and_val_fn([vid_in], targets, network)
    print('Compiled.')

    # Save pretrained net
    save_model(comb_filename, 'pretrain', network, log)

    # Precalc for announcing
    num_train_batches = int(np.ceil(float(len(dataset['train_labels']))/args.batch_size))
    num_test_batches = int(np.ceil(float(len(dataset['test_labels']))/args.batch_size))

    # Finally, launch the training loop.
    print("Starting training...")
    for epoch in range(args.num_epochs):
        print("Starting {} of {}.".format(epoch + 1, args.num_epochs))
        train_err = 0
        train_acc = 0
        train_batches = 0
        start_time = time.time()
        failed = 0

        # Call the data generator
        data_start_time = time.time()
        for data in d.flow(dataset, 'train', batch_size=args.batch_size, shuffle=True):
            data_prep_time = time.time() - data_start_time
            vid_in, vid_mask, bY = data
            # Do a training batch
            try:
                calc_start_time = time.time()
                err, acc = train_fn(vid_in, bY)
                calc_time = time.time() - calc_start_time
            except Exception as e:
                print(e)
                failed += 1
                print('Run failed!')
            train_err += err # Accumulate error
            train_acc += acc
            train_batches += 1 # Accumulate count so we can calculate mean later
            # Log and print
            log = store_in_log(log, {'b_train_err': err, 'b_train_acc' : acc})
            print("\tBatch {} of {} (FF: {:.2f}%): ".format(train_batches, num_train_batches,
                np.mean(vid_mask)*100.), end="")
            print("Loss: {:.3e} | Acc: {:2.2f}% | Data: {:.3f}s | Calc: {:.3f}s".format(
                float(err), acc*100., data_prep_time, calc_time))
            # Force it to go to output now rather than holding
            sys.stdout.flush()
            data_start_time = time.time()
        print("Training loss:\t\t{:.6f}".format(train_err / train_batches))

        # And a full pass over the validation data:
        val_err, val_acc = 0, 0
        val_batches = 0
        data_start_time = time.time()
        for data in d.flow(dataset, 'test', batch_size=args.batch_size, shuffle=False):
            data_prep_time = time.time() - data_start_time
            vid_in, vid_mask, bY = data
            try:
                calc_start_time = time.time()
                err, acc = val_fn(vid_in, bY)
                calc_time = time.time() - calc_start_time
            except Exception as e:
                print(e)
                failed+=1
                print('Run failed!')
            # Accumulate results
            val_err += err
            val_acc += acc
            val_batches += 1  # Accumulate count so we can calculate mean later
            # Log and print
            log = store_in_log(log, {'b_val_err': err, 'b_val_acc' : acc})
            print("\tBatch {} of {} (FF: {:.2f}%): ".format(val_batches, num_test_batches,
                np.mean(vid_mask)*100.), end="")
            print("Loss: {:.3e} | Acc: {:2.2f}% | Data: {:.3f}s | Calc: {:.3f}s".format(
                float(err), acc*100., data_prep_time, calc_time))
            sys.stdout.flush()
            data_start_time = time.time()

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, args.num_epochs, time.time() - start_time))
        # And we store
        log = store_in_log(log, {'val_err': val_err / val_batches,
                             'train_err': train_err / train_batches,
                             'val_acc':val_acc / val_batches*100.,
                             'train_acc':train_acc / train_batches*100.,} )

        print("\t Training loss:\t\t{:.6f}".format(log['train_err'][-1]))
        print("\t Validation loss:\t\t{:.6f}".format(log['val_err'][-1]))
        print("\t Training accuracy:\t\t{:.2f}".format(log['train_acc'][-1]))
        print("\t Validation accuracy:\t\t{:.2f}".format(log['val_acc'][-1]))
        print("\t Run failures: {}".format(failed))

        # Save result
        save_model(comb_filename, 'recent', network, log)

        # End if there's no improvement in validation error
        best_in_last_set = np.max(log[args.patience_key][-(args.patience-1):])
        # Drop out if our best round was not in the last set, i.e., no improvement
        if len(log[args.patience_key]) > args.wait_period and log[args.patience_key][-args.patience] >= best_in_last_set:
            break
        # Save best-so-far
        if log[args.patience_key][-1] >= np.max(log[args.patience_key]):
            save_model(comb_filename, 'best', network, log)

    # Save result
    save_model(comb_filename, 'final', network, log)
    print('Completed.')