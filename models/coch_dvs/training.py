
# coding: utf-8

# In[5]:

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
from lipreading_utils_grad import pad_sequences,find_data_mean, find_data_std, HDF5LipReadingIterator, HDF5LipReadingIterator_events,HDF5LipReadingIterator_sep,ftensor5, get_train_and_val_fn, vocab_size, print_model
from lasagne_utils import save_model, store_in_log, load_model, load_log, replace_updates_nans_with_zero,load_branch_param
from lipreading_models import get_dan_original_rnn,get_network,get_network_12classes,get_dan_original_video,get_video_coch_net
import h5py

T.config.floatX='float32'


# In[6]:

#
# In[ ]:

# Set seed
np.random.seed(12)
patience_key='val_acc'
patience=5
wait_period=20
#layer indices of audio/video in the joint network
#parameter indices of audio/video in the saved log file
coch_spike_para_ind=10
vid_frame_para_ind=16
vid_layer_ind=14
aud_layer_ind=16
# parameters/args
#directories for audio and video data, saved as .npy
#aud_data_dir='/home/yy/thesis/data/sentence_tests_5_18/np_data/aud_data/'
#vid_data_dir='/home/yy/thesis/data/sentence_tests_5_18/np_data/vid_data/'
h5file_video = '/home/xiaoya/lipreading/DVS_corr.hdf5'
h5file_coch='/home/xiaoya/lipreading/grid_coch_ctime_40_on3_normed2.hdf5'

test_run_id='0'
# Get dataset
#log = load_log(filename)

# Load dataset
dataset_video=h5py.File(h5file_video, "r")
dataset_coch = h5py.File(h5file_coch, "r")

print('Train set size: {} sentences.'.format(len(dataset_video['train_labels'])))
print('Test set size: {} sentences.'.format(len(dataset_video['test_labels'])))

# Load network
aud_in   = T.ftensor3('aud_in')
vid_in   = ftensor5('vid_in')
targets  = T.fmatrix('targets')
network = get_video_coch_net(aud_in, vid_in)
#load_model(filename, network)

# Get data iterator
d = HDF5LipReadingIterator_sep()

# Compile the output fn
print('Compiling output functions...')
train_fn, val_fn, out_fn = get_train_and_val_fn([aud_in, vid_in], targets, network)
print('Compiled.')

print('Data prepped and passed through.')
#Probability of losing either an audio or video stream
test_blankout=0.0
batch_size=100




# In[ ]:

filename = 'lipreading_grid_dvs_coch_corr'
new_filename=filename+'_0'
vid_model_filename='lipreading_grid_video_48_3_best'
aud_model_filename='lipreading_grid_coch_ctime_40_on_0_best'
oldfilename=filename+'_2_final'

blankout=0.5
aud_blankout=0.8

# Create symbolic vars
#aud_in   = T.ftensor3('aud_in')
aud_mask = T.bmatrix('aud_mask')
#vid_in   = ftensor5('vid_in')
vid_mask = T.bmatrix('vid_mask')
#targets  = T.fmatrix('targets')
#print(aud_in.type)
#print(aud_mask.type)
#print(vid_in.type)
#print(vid_mask.type)
#print(T.config.floatX)
   
# Build model
print("Building network ...")
#   Get input dimensions
#network = get_video_coch_net(aud_in, vid_in)
# Instantiate log
log = defaultdict(list)
print("Built.")
    

# Dump some debug data if we like
#print_model(network)
#load_branch_param(vid_model_filename, network,vid_frame_para_ind,vid_layer_ind)
#load_branch_param(aud_model_filename, network,coch_spike_para_ind,aud_layer_ind)
#load_model(oldfilename,network)
# Compile the learning functions
#print('Compiling functions...')
#train_fn, val_fn, out_fn = get_train_and_val_fn([aud_in, vid_in], targets, network)
#print('Compiled.')

# Save pretrained net
save_model(new_filename, 'pretrain', network, log)


# Precalc for announcing
num_train_batches = int(np.ceil(float(len(dataset_coch['train_labels']))/batch_size))
num_test_batches = int(np.ceil(float(len(dataset_coch['test_labels']))/batch_size))

# pass testing data to model
val_err, val_acc = 0, 0
debug_var,debug_var2=0,0

val_no_aud_err, val_no_aud_acc = 0, 0
val_no_vid_err, val_no_vid_acc = 0, 0
val_batches = 0
data_start_time = time.time()
num_epochs=200
# Finally, launch the training loop.
print("Starting training...")
for epoch in range(num_epochs):
    print("Starting {} of {}.".format(epoch + 1, num_epochs))
    train_err = 0
    train_acc = 0
    train_err_no_aud=0
    train_acc_no_aud=0
    train_err_no_vid=0
    train_acc_no_vid=0
    train_batches = 0
    grads=[]

    start_time = time.time()
    failed = 0

    # Call the data generator
    data_start_time = time.time()
    for data in d.flow(dataset_video,dataset_coch, 'train', batch_size=batch_size, shuffle=True, blankout=blankout,
            audio_blankout_prob=aud_blankout):
        data_prep_time = time.time() - data_start_time
        aud_in, aud_mask, vid_in, vid_mask, bY = data
	
        print(aud_in.shape)
        print(vid_in.shape)
        print(bY.shape)
        if aud_in.shape[1]>200:
		continue

	

	# Do a training batch
        try:
            calc_start_time = time.time()
            err, acc,debug_var,grads_normed,actvivations = train_fn(aud_in,vid_in, bY)
            no_aud_err_train,no_aud_acc_train,_ = val_fn(aud_in*0., vid_in, bY)
            no_vid_err_train,no_vid_acc_train,_ = val_fn(aud_in, vid_in*0., bY)

            calc_time = time.time() - calc_start_time
        except Exception as e:
            print(e)
            failed += 1
            print('Run failed!')
        train_err += err # Accumulate error
        train_acc += acc
        train_err_no_aud+=no_aud_err_train
        train_acc_no_aud+=no_aud_acc_train
        train_err_no_vid+=no_vid_err_train
        train_acc_no_vid+=no_vid_acc_train

        grads.append(grads_normed)
        train_batches += 1 # Accumulate count so we can calculate mean later
        # Log and print
        log = store_in_log(log, {'b_train_err': err, 'b_train_acc' : acc, 'b_train_err_no_aud':no_aud_err_train,'b_train_acc_no_aud':no_aud_acc_train,
            'b_train_err_no_vid':no_vid_err_train,'b_train_acc_no_vid':no_vid_acc_train,
            'b_grads_normed' : grads_normed,'b_activations':actvivations})
        print("\tBatch {} of {} (FF: {:.2f}%): ".format(train_batches, num_train_batches,
            np.mean(aud_mask)*100.), end="")        
        print("Loss: {:.3e} | Acc: {:2.2f}% | Data: {:.3f}s | Calc: {:.3f}s".format(
            float(err), acc*100., data_prep_time, calc_time))
        #print(debug_var)
        # Force it to go to output now rather than holding
        sys.stdout.flush()
        data_start_time = time.time()
    print("Training loss:\t\t{:.6f}".format(train_err / train_batches))

    # And a full pass over the validation data:
    val_err, val_acc = 0, 0
    val_no_aud_err, val_no_aud_acc = 0, 0
    val_no_vid_err, val_no_vid_acc = 0, 0
    val_batches = 0
    data_start_time = time.time()
    for data in d.flow(dataset_video,dataset_coch, 'test', batch_size=batch_size, shuffle=False, blankout=test_blankout):
        aud_in, aud_mask, vid_in, vid_mask, bY = data
        print(bY.shape)
        try:
            calc_start_time = time.time()
            err, acc,debug_var2 = val_fn(aud_in,vid_in,  bY)
            no_aud_err, no_aud_acc,_= val_fn(aud_in*0., vid_in, bY)
            no_vid_err, no_vid_acc,_ = val_fn(aud_in, vid_in*0., bY)
            calc_time = time.time() - calc_start_time
        except Exception as e:
            print(e)
            failed+=1
            print('Run failed!')
        # Accumulate results
        val_err += err
        val_acc += acc
        val_no_vid_err += no_vid_err
        val_no_vid_acc += no_vid_acc
        val_no_aud_err += no_aud_err
        val_no_aud_acc += no_aud_acc
        val_batches += 1  # Accumulate count so we can calculate mean later
        print
        # Log and print
        log = store_in_log(log, {'b_val_err': err, 'b_val_acc' : acc
                                })
        print("\tBatch {} of {} (FF: {:.2f}%): ".format(val_batches, num_test_batches,
            np.mean(aud_mask)*100.), end="")
        print("Loss: {:.3e} | Acc: {:2.2f}% | Data: {:.3f}s | Calc: {:.3f}s".format(
            float(err), acc*100., data_prep_time, calc_time))
        #print(debug_var2)
        sys.stdout.flush()
        data_start_time = time.time()

    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))
    # And we store
    log = store_in_log(log, {'val_err': val_err / val_batches,
                             'train_err': train_err / train_batches,
                             'val_acc':val_acc / val_batches*100.,
                             'train_acc':train_acc / train_batches*100.,
                             'val_no_vid_acc':val_no_vid_acc/val_batches,
                             'val_no_aud_acc':val_no_aud_acc/val_batches,
                             'train_no_aud_acc':train_acc_no_aud/train_batches,
                             'train_no_vid_acc':train_acc_no_vid/train_batches,
                 'grads' : sum(grads)
                             } )

    print("\t Training loss:\t\t{:.6f}".format(log['train_err'][-1]))
    print("\t Validation loss:\t\t{:.6f}".format(log['val_err'][-1]))
    print("\t Training accuracy:\t\t{:.2f}".format(log['train_acc'][-1]))
    print("\t Validation accuracy:\t\t{:.2f}".format(log['val_acc'][-1]))
    print("\t Validation accuracy(no video):\t\t{:.2f}".format(log['val_no_vid_acc'][-1]))
    print("\t Validation accuracy(no audio):\t\t{:.2f}".format(log['val_no_aud_acc'][-1]))
    print("\t Training accuracy(no audio):\t\t{:.2f}".format(log['train_no_aud_acc'][-1]))
    print("\t Training accuracy(no video):\t\t{:.2f}".format(log['train_no_vid_acc'][-1]))
    print("\t Run failures: {}".format(failed))
    
    # Save result
    save_model(new_filename, 'recent', network, log)
    
    # End if there's no improvement in validation error
    #best_in_last_set = np.max(log[patience_key][-(patience-1):])
    # Drop out if our best round was not in the last set, i.e., no improvement
    #if len(log[patience_key]) > wait_period and log[patience_key][-patience] >= best_in_last_set:
    #    break
    # Save best-so-far
    if log[patience_key][-1] >= np.max(log[patience_key]):
        save_model(new_filename, 'best', network, log)

    
# Save result
save_model(new_filename, 'final', network, log)
print('Completed.')



