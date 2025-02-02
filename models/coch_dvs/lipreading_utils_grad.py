import numpy as np
from collections import defaultdict
import theano.tensor as T
import lasagne
import theano
import time
from lasagne.regularization import regularize_layer_params, l2, l1
from PIL import Image
import pickle

# Create 5D tensor type
ftensor5 = T.TensorType(dtype="float32", broadcastable=(False,) * 5)

# Build vocabulary
commands = ['bin', 'lay', 'place', 'set']
colors = ['blue', 'green', 'red', 'white']
prepositions = ['at','by','in','with']
letters = list(set(map(chr, range(ord('a'), ord('z')+1))) - set('w'))
digits = [str(i) for i in range(1,10)] + ['zero']
#digits = [str(i) for i in range(0,10)]
adverbs = ['again','now','please','soon']

# Build translation by first letter
tr_commands = {item[0]:item for item in commands}
tr_colors = {item[0]:item for item in colors}
tr_prepositions = {item[0]:item for item in prepositions}
tr_letters = {item[0]:item for item in letters}
tr_digits = {item[0]:item for item in digits}
tr_adverbs = {item[0]:item for item in adverbs}

# Build vocabulary translation dict
all_vocab = commands + colors + prepositions + letters + digits + adverbs
#all_vocab = commands + colors + prepositions
word_to_idx = {vocab:idx for idx, vocab in enumerate(all_vocab)}
idx_to_word = {value:key for key,value in word_to_idx.items()}
vocab_size = len(all_vocab)

vid_layer_ind=14
aud_layer_ind=16

from lasagne.updates import get_or_compute_grads, utils
from collections import OrderedDict
def adam_w_steps_too(loss_or_grads, params, learning_rate=0.001, beta1=0.9,
         beta2=0.999, epsilon=1e-8):
    """    
    Adam rewrite that returns step updates as well as the actual replaced value.
    """
    all_grads = get_or_compute_grads(loss_or_grads, params)
    t_prev = theano.shared(utils.floatX(0.))
    updates = OrderedDict()
    # DAN: New:
    steps = OrderedDict()
    
    # Using theano constant to prevent upcasting of float32
    one = T.constant(1)

    t = t_prev + 1
    a_t = learning_rate*T.sqrt(one-beta2**t)/(one-beta1**t)

    for param, g_t in zip(params, all_grads):
        value = param.get_value(borrow=True)
        m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)
        v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)

        m_t = beta1*m_prev + (one-beta1)*g_t
        v_t = beta2*v_prev + (one-beta2)*g_t**2
        step = a_t*m_t/(T.sqrt(v_t) + epsilon)

        updates[m_prev] = m_t
        updates[v_prev] = v_t
        updates[param] = param - step
        # DAN: New:
        steps[param] = step

    updates[t_prev] = t
    return updates, steps


def bow_to_said_list(bow):
    bow_idxs = np.argwhere(bow==1).flatten()
    out_of_order = [idx_to_word[idx] for idx in bow_idxs]
    # sort_order = [display_order.index(word) for word in out_of_order]
    return out_of_order
    # return [word for (order, word) in sorted(zip(sort_order, out_of_order))]

def find_data_mean(keys, AUD_DATA_DIR, VID_DATA_DIR):
    tot_idxs = len(keys)
    tot_count = 0
    # Find video mean
    curr_vid_data = np.load(VID_DATA_DIR + keys[0] + '_vid.npy')
    mean_vid = np.zeros( (curr_vid_data.shape[1], curr_vid_data.shape[2], 1) )
    for key in keys:
        curr_vid_data = np.load(VID_DATA_DIR + key + '_vid.npy')
        # Make grayscale
        curr_vid_data = np.expand_dims(curr_vid_data.mean(axis=3), axis=3)
        mean_vid += np.sum(curr_vid_data, axis=0)
        tot_count += curr_vid_data.shape[0]
    mean_vid = mean_vid / tot_count
    # Find audio mean
    tot_count = 0
    curr_aud_data = np.load(AUD_DATA_DIR + keys[0] + '_aud.npy')
    mean_aud = np.zeros( curr_aud_data.shape[1] )
    for key in keys:
        try:
            curr_aud_data = np.load(AUD_DATA_DIR + key + '_aud.npy')
            mean_aud += np.sum(curr_aud_data, axis=0)
            tot_count += curr_aud_data.shape[0]
        except Exception as e:
            pass
    mean_aud = mean_aud / tot_count
    return mean_aud, mean_vid

def find_data_std(keys, mean_aud, mean_vid, AUD_DATA_DIR, VID_DATA_DIR):
    tot_idxs = len(keys)
    tot_count = 0
    # Find video mean
    curr_vid_data = np.load(VID_DATA_DIR + keys[0] + '_vid.npy')
    sum_sq_vid = np.zeros( (curr_vid_data.shape[1], curr_vid_data.shape[2], 1) )
    for key in keys:
        curr_vid_data = np.load(VID_DATA_DIR + key + '_vid.npy')
        # Make grayscale
        curr_vid_data = np.expand_dims(curr_vid_data.mean(axis=3), axis=3)
        sum_sq_vid += np.sum((curr_vid_data-mean_vid)**2, axis=0)
        tot_count += curr_vid_data.shape[0]
    std_vid = np.sqrt(sum_sq_vid / tot_count)
    # Find audio mean
    tot_count = 0
    curr_aud_data = np.load(AUD_DATA_DIR + keys[0] + '_aud.npy')
    sum_sq_aud = np.zeros( curr_aud_data.shape[1] )
    for key in keys:
        try:
            curr_aud_data = np.load(AUD_DATA_DIR + key + '_aud.npy')
            sum_sq_aud += np.sum((curr_aud_data-mean_aud)**2, axis=0)
            tot_count += curr_aud_data.shape[0]
        except Exception as e:
            pass
    std_aud = np.sqrt(sum_sq_aud / tot_count)
    return std_aud, std_vid

def translate_short_string(short_string):
    assert len(short_string)==6, 'Input should be short form of the sentence.'
    command =     tr_commands[short_string[0]]
    color   =       tr_colors[short_string[1]]
    prep    = tr_prepositions[short_string[2]]
    letter  =      tr_letters[short_string[3]]
    digit   =       tr_digits[short_string[4]]
    adverb  =      tr_adverbs[short_string[5]]
    return [command, color, prep, letter, digit, adverb]

def translate_short_string_12classes(short_string):
    assert len(short_string)==3, 'Input should be short form of the sentence.'
    command =     tr_commands[short_string[0]]
    color   =       tr_colors[short_string[1]]
    prep    = tr_prepositions[short_string[2]]
    return [command, color, prep]

def get_data_from_key(key, mean_feats, std_feats, data_dirs):
    curr_aud_data = np.load(data_dirs[0] + key + '_aud.npy')
    curr_aud_data = (curr_aud_data - mean_feats[0]) / std_feats[0]
    curr_vid_data = np.transpose(np.load(data_dirs[1] + key + '_vid.npy'), axes=(0,3,1,2))
    # Grayscale, and reinsert the lost dimension: 75x1x48x48
    curr_vid_data = np.expand_dims(curr_vid_data.mean(axis=1), axis=1)
    curr_vid_data = ((curr_vid_data - mean_feats[1]) / std_feats[1])
    return curr_aud_data, curr_vid_data

def set_indices_to_one(vec, indices):
    vec[indices] = 1.
    #vec[-1] = 1. #dummy class
    return vec

def short_sentence_to_bow(short_string):
    bow = [word_to_idx[word] for word in translate_short_string(short_string)]
    return set_indices_to_one(np.zeros(vocab_size), bow)

def short_sentence_to_bow_12classes(short_string):
    bow = [word_to_idx[word] for word in translate_short_string_12classes(short_string)]
    return set_indices_to_one(np.zeros(vocab_size+1), bow)

def pad_and_prep_datapair(aud_data, vid_data):
    max_len = 0
    max_len = np.max([max_len] + [item.shape[0] for item in aud_data])
    max_len = np.max([max_len] + [item.shape[0] for item in vid_data])

    aud_data, aud_mask = pad_sequences(aud_data, max_len)
    vid_data, vid_mask = pad_sequences(vid_data, max_len)

    return aud_data, aud_mask, vid_data, vid_mask

# Define a function to zero-pad data
def pad_sequences(sequences, max_len, dtype='float32', padding='pre', truncating='pre', value=0.):
    # (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)
    nb_samples = len(sequences)
    # Check for 2D Audio Histograms or 4D Videos
    if len(sequences[0].shape) == 2:
        x = (np.ones((nb_samples, max_len, sequences[0].shape[1])) * value).astype(dtype)
        mask = (np.ones((nb_samples, max_len)) * value).astype('bool')
    elif len(sequences[0].shape) == 4:
        x = (np.ones((nb_samples, max_len,
                      sequences[0].shape[1], sequences[0].shape[2], sequences[0].shape[3])) * value).astype(dtype)
        mask = (np.ones((nb_samples, max_len)) * value).astype('bool')
    # Loop through and pad
    for idx, s in enumerate(sequences):
        if truncating == 'pre':
            trunc = s[ -max_len:]
        elif truncating == 'post':
            trunc = s[:max_len]
        if padding == 'post':
            x[idx, :len(trunc)] = trunc
            mask[idx, :len(trunc)] = 1
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
            mask[idx, -len(trunc):] = 1
    return x, mask

class LipReadingIterator(object):
    def flow(self, data_keys, mean_feats, std_feats, data_dirs, batch_size, shuffle=True, blankout=0, flatten_color=True):
        # Get some constants
        num_examples = len(data_keys)
        num_batches = int(np.ceil(float(num_examples)/batch_size))
        # Shuffle the data
        if shuffle:
            np.random.shuffle(data_keys)
        b = 0
        while b < num_batches:
            curr_keys = data_keys[b*batch_size:(b+1)*batch_size]
            aud_data, vid_data, bY = [], [], []
            for key in curr_keys:
                curr_aud_data, curr_vid_data = get_data_from_key(key, mean_feats, std_feats, data_dirs)
                # Check to disable one of the streams
                if blankout>0 and np.random.rand()<blankout:
                    if np.random.rand()<0.8:
                        aud_data.append(curr_aud_data*0.)
                        vid_data.append(curr_vid_data)
                    else:
                        aud_data.append(curr_aud_data)
                        vid_data.append(curr_vid_data*0.)
                else:
                    aud_data.append(curr_aud_data)
                    vid_data.append(curr_vid_data)
                bY.append(short_sentence_to_bow(key.split('-')[1]))
            prepped_data = pad_and_prep_datapair(aud_data, vid_data)
            # Plop the target Ys on the end
            yield list(prepped_data) + [np.array(bY).astype('float32')]
            b += 1

def pad_datastreams(datastreams, rand_pad=10):
    num_streams = len(datastreams)
    #print(num_streams)
    max_len = np.max(np.max([[len(item) for item in datastream] for datastream in datastreams]))
    #print(max_len)
    max_len = max_len + rand_pad
    returned_streams = []
    for datastream in datastreams:
        num_items = len(datastream)
        #print(num_items)
        # First dim is batch size, second is length (going to be overwritten to max)
        size_tuple = datastream[0].shape[1:]
        data_tensor = np.zeros( (num_items, max_len) + size_tuple, dtype='float32')
        mask_tensor = np.zeros( (num_items, max_len), dtype='float32')
        for idx, item in enumerate(datastream):
            #start_offset = np.random.randint(low=0, high=max_len-len(item))
            start_offset = 0
            data_tensor[idx, start_offset:start_offset+len(item)] = item
            mask_tensor[idx, start_offset:start_offset+len(item)] = 1.
        returned_streams.append(data_tensor)
        returned_streams.append(mask_tensor)
    return returned_streams

def pad_datastreams_rand(datastreams, rand_pad=20):
    num_streams = len(datastreams)
    #print(num_streams)
    max_len = np.max(np.max([[len(item) for item in datastream] for datastream in datastreams]))
    #print(max_len)
    max_len = max_len + rand_pad
    returned_streams = []
    for datastream in datastreams:
        num_items = len(datastream)
        #print(num_items)
        # First dim is batch size, second is length (going to be overwritten to max)
        size_tuple = datastream[0].shape[1:]
        data_tensor = np.zeros( (num_items, max_len) + size_tuple, dtype='float32')
        mask_tensor = np.zeros( (num_items, max_len), dtype='float32')
        for idx, item in enumerate(datastream):
            start_offset = np.random.randint(low=0, high=max_len-len(item))
            #start_offset = 0
            data_tensor[idx, start_offset:start_offset+len(item)] = item
            mask_tensor[idx, start_offset:start_offset+len(item)] = 1.
        returned_streams.append(data_tensor)
        returned_streams.append(mask_tensor)
    return returned_streams



def get_hdf5_aud_data(data_collection, datagroup_key, key, mean_aud, std_aud):
    aud_data = (np.array(data_collection[datagroup_key+'_aud'][key]) - mean_aud)/std_aud
    return aud_data

def get_hdf5_vid_data(data_collection, datagroup_key, key, mean_vid, std_vid, decimate=1):
    vid_data = np.expand_dims((np.mean(data_collection[datagroup_key+'_vid'][key][::decimate], axis=3) - mean_vid)/std_vid, axis=1)
    #vid_data = np.expand_dims((data_collection[datagroup_key+'_vid'][key][::decimate] - mean_vid)/std_vid, axis=1)
    return vid_data

def get_hdf5_coch_data(data_collection, datagroup_key, key):
    aud_data = np.transpose(np.array(data_collection[datagroup_key+'_coch'][key]))
    return aud_data

def resize_vid(vid,final_size=(48,48)):
    resized_vid=[]
    if vid.shape[0]==0:
	return resized_vid
    for i in range(vid.shape[0]):
        im = Image.fromarray(vid[i,:,:])
        im.thumbnail(final_size, Image.ANTIALIAS)
        resized = np.array(im.getdata(),np.uint8).reshape(
        im.size[0], im.size[1])
    resized_vid.append(resized)
    return resized_vid
    

def get_hdf5_dvs_data(data_collection, datagroup_key, key, decimate=1):
    #vid_data = np.expand_dims((np.mean(data_collection[datagroup_key+'_vid'][key][::decimate], axis=3) - mean_vid)/std_vid, axis=1)
    #vid_data=resize_vid(data_collection[datagroup_key+'_dvs'][key][::decimate])
    vid_data=data_collection[datagroup_key+'_dvs'][key][::decimate]
    vid_data=np.transpose(vid_data, (2,0,1))
    vid_data = np.expand_dims(vid_data, axis=1)
    
    return vid_data

class HDF5LipReadingIterator_events(object):
    def flow(self, data_collection, datagroup_key, batch_size, shuffle=True, blankout=0, audio_blankout_prob=0.0):
        # Get some constants
        #no normalization
        #mean_aud, std_aud = data_collection['mean_aud'], data_collection['std_aud']
        #mean_vid, std_vid = data_collection['mean_vid'], data_collection['std_vid']
        data_keys = np.array(data_collection[datagroup_key+'_labels'])
        num_examples = len(data_keys)
        num_batches = int(np.ceil(float(num_examples)/batch_size))
        # Shuffle the data
        if shuffle:
            np.random.shuffle(data_keys)
        b = 0
        while b < num_batches:
            curr_keys = data_keys[b*batch_size:(b+1)*batch_size]
            aud_data, vid_data, bY = [], [], []
            for key in curr_keys:
                curr_aud_data = get_hdf5_coch_data(data_collection, datagroup_key, key)
                curr_vid_data = get_hdf5_dvs_data(data_collection, datagroup_key, key)
                # Check to disable one of the streams
                if blankout>0 and np.random.rand()<blankout:
                    if np.random.rand()<audio_blankout_prob:
                        aud_data.append(np.zeros_like(curr_aud_data))
                        vid_data.append(curr_vid_data)
                    else:
                        aud_data.append(curr_aud_data)
                        vid_data.append(np.zeros_like(curr_vid_data))
                else:
                    aud_data.append(curr_aud_data)
                    vid_data.append(curr_vid_data)
                bY.append(short_sentence_to_bow(key.split('-')[1]))
                #bY.append(short_sentence_to_bow(key[3:]))
            aud, mask_aud, vid, mask_vid = pad_datastreams([np.array(aud_data), np.array(vid_data)])
            yield [aud.astype('float32'), mask_aud.astype('float32'), vid.astype('float32'), mask_vid.astype('float32'), np.array(bY).astype('float32')]
            b += 1
            
class HDF5LipReadingIterator(object):
    def flow(self, data_collection, datagroup_key, batch_size, shuffle=True, blankout=0, audio_blankout_prob=0.8):
        # Get some constants
        #mean_aud, std_aud = data_collection['mean_aud'], data_collection['std_aud']
        mean_vid, std_vid = data_collection['mean_vid'], data_collection['std_vid']
        mean_vid=np.squeeze(mean_vid)
        std_vid=np.squeeze(std_vid)
        data_keys = np.array(data_collection[datagroup_key+'_labels'])
        num_examples = len(data_keys)
        num_batches = int(np.ceil(float(num_examples)/batch_size))
        # Shuffle the data
        if shuffle:
            np.random.shuffle(data_keys)
        b = 0
        while b < num_batches:
            curr_keys = data_keys[b*batch_size:(b+1)*batch_size]
            aud_data, vid_data, bY = [], [], []
            for key in curr_keys:
                #curr_aud_data = get_hdf5_aud_data(data_collection, datagroup_key, key, mean_aud, std_aud)
                curr_aud_data = get_hdf5_coch_data(data_collection, datagroup_key, key)
                curr_vid_data = get_hdf5_vid_data(data_collection, datagroup_key, key, mean_vid , std_vid)
                # Check to disable one of the streams
                if blankout>0 and np.random.rand()<blankout:
                    if np.random.rand()<audio_blankout_prob:
                        aud_data.append(np.zeros_like(curr_aud_data))
                        vid_data.append(curr_vid_data)
                    else:
                        aud_data.append(curr_aud_data)
                        vid_data.append(np.zeros_like(curr_vid_data))
                else:
                    aud_data.append(curr_aud_data)
                    vid_data.append(curr_vid_data)
                bY.append(short_sentence_to_bow(key.split('-')[1]))
                #bY.append(short_sentence_to_bow_12classes(key[0:3]))
            aud, mask_aud, vid, mask_vid = pad_datastreams([np.array(aud_data), np.array(vid_data)])
            #vid, mask_vid = pad_datastreams([np.array(vid_data)])
            #yield [aud.astype('float32'), mask_aud.astype('float32'), vid.astype('float32'), mask_vid.astype('float32'), np.array(bY).astype('float32')]
            yield [aud.astype('float32'), mask_aud.astype('float32'),vid.astype('float32'), mask_vid.astype('float32'), np.array(bY).astype('float32')]
            b += 1

class HDF5LipReadingIterator_shift(object):
    def flow(self, data_collection_video, data_collection_audio, datagroup_key, batch_size, shifts_per_sample, shuffle=True, blankout=0, audio_blankout_prob=0.8):
        group_keys=['train','test']
        ind=group_keys.index(datagroup_key)
        the_other_key=group_keys[1-ind]
        # Get some constants
        #mean_aud, std_aud = data_collection_audio['mean_aud'], data_collection_audio['std_aud']
        #mean_vid, std_vid = data_collection_video['mean_vid'], data_collection_video['std_vid']
        #mean_vid=np.squeeze(mean_vid)
        #std_vid=np.squeeze(std_vid)
        data_keys = np.array(data_collection_video[datagroup_key+'_labels'])
        
        num_examples = len(data_keys)
        num_batches = int(np.ceil(float(num_examples)/batch_size))

        # Shuffle the data
        if shuffle:
            np.random.shuffle(data_keys)
        b = 0
        while b < num_batches:

            curr_keys = data_keys[b*batch_size:(b+1)*batch_size]
            aud_data, vid_data, bY = [], [], []

            for key in curr_keys:
                label=key.split('-')[1]
                curr_aud_data,curr_vid_data=[],[]
            
		#curr_aud_data = get_hdf5_coch_data(data_collection_audio, datagroup_key, key)
                try:
		    curr_vid_data = get_hdf5_dvs_data(data_collection_video, datagroup_key, key)
		except Exception as e:
		    print('unable to load video sample')
		    continue               

                try:
                    curr_aud_data = get_hdf5_coch_data(data_collection_audio, datagroup_key, key)
                except Exception as e:    
                    try:
                        curr_aud_data = get_hdf5_coch_data(data_collection_audio, the_other_key, key)
                       
                    except Exception as e:
                        continue

                for s in range(shifts_per_sample):       
                    # Check to disable one of the streams
                    if blankout>0 and np.random.rand()<blankout:
                        if np.random.rand()<audio_blankout_prob:
                            aud_data.append(np.zeros_like(curr_aud_data))
                            vid_data.append(curr_vid_data)
                        else:
                            aud_data.append(curr_aud_data)
                            vid_data.append(np.zeros_like(curr_vid_data))
                    else:
                        aud_data.append(curr_aud_data)
                        vid_data.append(curr_vid_data)
                    bY.append(short_sentence_to_bow(label))

                #bY.append(short_sentence_to_bow_12classes(key[0:3]))
            if len(aud_data)==0:
                b += 1
                continue
            aud, mask_aud, vid, mask_vid = pad_datastreams_rand([np.array(aud_data), np.array(vid_data)])
            #vid, mask_vid = pad_datastreams([np.array(vid_data)])
            yield [aud.astype('float32'), mask_aud.astype('float32'), vid.astype('float32'), mask_vid.astype('float32'), np.array(bY).astype('float32')]
            #yield [vid.astype('float32'), mask_vid.astype('float32'), np.array(bY).astype('float32')]
            b += 1

class HDF5LipReadingIterator_sep(object):
    def flow(self, data_collection_video, data_collection_audio, datagroup_key, batch_size, shuffle=True, blankout=0, audio_blankout_prob=0.8):
        group_keys=['train','test']
        ind=group_keys.index(datagroup_key)
        the_other_key=group_keys[1-ind]
        # Get some constants
        #mean_aud, std_aud = data_collection_audio['mean_aud'], data_collection_audio['std_aud']
        #mean_vid, std_vid = data_collection_video['mean_vid'], data_collection_video['std_vid']
        #mean_vid=np.squeeze(mean_vid)
        #std_vid=np.squeeze(std_vid)
        data_keys = np.array(data_collection_video[datagroup_key+'_labels'])
        
        num_examples = len(data_keys)
        num_batches = int(np.ceil(float(num_examples)/batch_size))
        file_cnt=0
        # Shuffle the data
        if shuffle:
            np.random.shuffle(data_keys)
        b = 0
        while b < num_batches:
            curr_keys = data_keys[b*batch_size:(b+1)*batch_size]
            aud_data, vid_data, bY = [], [], []
            for key in curr_keys:
                curr_aud_data,curr_vid_data=[],[]
                #curr_aud_data = get_hdf5_coch_data(data_collection_audio, datagroup_key, key)
                curr_vid_data = get_hdf5_dvs_data(data_collection_video, datagroup_key, key)
                
                f1=open('checked_vid_keys','a')
                f1.write(key+'\n')
                f1.close()

                file_cnt=file_cnt+1
                try:
                    curr_aud_data = get_hdf5_coch_data(data_collection_audio, datagroup_key, key)
                    
                    f2=open('checked_aud_keys','a')
                    f2.write(key+'\n')
                    f2.close()
                except Exception as e:    
                    try:
                        curr_aud_data = get_hdf5_coch_data(data_collection_audio, the_other_key, key)
                        f2=open('checked_aud_keys','a')
                        f2.write(key+'\n')
                        f2.close()
                    except Exception as e:
                        continue

                    
                # Check to disable one of the streams
                if blankout>0 and np.random.rand()<blankout:
                    if np.random.rand()<audio_blankout_prob:
                        aud_data.append(np.zeros_like(curr_aud_data))
                        vid_data.append(curr_vid_data)
                    else:
                        aud_data.append(curr_aud_data)
                        vid_data.append(np.zeros_like(curr_vid_data))
                else:
                    aud_data.append(curr_aud_data)
                    vid_data.append(curr_vid_data)
                bY.append(short_sentence_to_bow(key.split('-')[1]))
                f3=open('used_keys','a')
                f3.write(key+'\n')
                f3.close()
                #bY.append(short_sentence_to_bow_12classes(key[0:3]))
            if len(aud_data)==0:
                b += 1
                continue
            aud, mask_aud, vid, mask_vid = pad_datastreams([np.array(aud_data), np.array(vid_data)])
            #vid, mask_vid = pad_datastreams([np.array(vid_data)])
            yield [aud.astype('float32'), mask_aud.astype('float32'), vid.astype('float32'), mask_vid.astype('float32'), np.array(bY).astype('float32')]
            #yield [vid.astype('float32'), mask_vid.astype('float32'), np.array(bY).astype('float32')]
            b += 1
        print(file_cnt)
            
def print_model(model):
    print('All parameters:')
    for layer_idx, layer in enumerate(lasagne.layers.get_all_layers(model)):
        print('Layer {: >2}: {}'.format(layer_idx, layer.__class__))
        for param, options in layer.params.items():
            print('\t\t{}: Size: {}'.format(param, param.get_value().shape))

def print_model_with_data(model, data):
    print('All parameters:')
    for layer_idx, layer in enumerate(lasagne.layers.get_all_layers(model)):
        print('Layer {: >2}: {}'.format(layer_idx, layer.__class__))
        for param, options in layer.params.items():
            print('\t\t{}: Size: {}'.format(param, param.get_value().shape))
        print('\t\tOutput data shape: {}'.format(data[layer_idx].shape))

def get_layer_output_fn(fn_inputs, network):
    outs = []
    for layer in lasagne.layers.get_all_layers(network):
        outs.append(lasagne.layers.get_output(layer, deterministic=True))
    out_fn = theano.function(fn_inputs, outs)
    return out_fn

def get_train_and_val_fn(inputs, target_var, network):
    prediction = lasagne.layers.get_output(network)
    # Complicated math to get the mean per-example error rate
    prediction_thresholded = T.switch(prediction>0.5, 1.0, 0.0)
    sum_of_targets = T.sum(target_var, axis=1)
    mean_mismatch_per_example = T.sum(abs(prediction_thresholded-target_var), axis=1)/sum_of_targets
    train_acc = 1.-T.mean(mean_mismatch_per_example, dtype=theano.config.floatX)
    loss = lasagne.objectives.squared_error(prediction, target_var)
    #l2 regularization
    layers=lasagne.layers.get_all_layers(network)
    l2_penalty = regularize_layer_params(layers, l2)
    
    loss = loss.mean()#+l2_penalty*1e-5

    #params = lasagne.layers.get_all_params(network, trainable=True)
    #updates = lasagne.updates.adam(loss, params, learning_rate=1e-3)
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates, steps = adam_w_steps_too(loss, params,learning_rate=5e-4)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    # Complicated math to get the mean per-example error rate
    test_prediction_thresholded = T.switch(test_prediction>0.5, 1.0, 0.0)
    sum_of_targets = T.sum(target_var, axis=1)
    test_mean_mismatch_per_example = T.sum(abs(test_prediction_thresholded-target_var), axis=1)/sum_of_targets
    test_acc = 1.-T.mean(test_mean_mismatch_per_example, dtype=theano.config.floatX)
    test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
    test_loss = test_loss.mean()

    vid_activation = T.mean(T.sum(T.sum(lasagne.layers.get_output(layers[vid_layer_ind], deterministic=True),axis=1)))
    aud_activation = T.mean(T.sum(T.sum(lasagne.layers.get_output(layers[aud_layer_ind], deterministic=True),axis=1)))
    post_merge_activation=T.mean(T.sum(T.sum(lasagne.layers.get_output(layers[-2], deterministic=True),axis=1)))

    activations = [vid_activation,aud_activation,post_merge_activation]
    activations=T.as_tensor_variable(activations)
    out_fn = get_layer_output_fn(inputs, network)
    fn_inputs = inputs + [target_var]

    # Get gradients
    # Create a list of parameters by layer
    param_layer_lookup = {}
    for layer_idx, layer in enumerate(lasagne.layers.get_all_layers(network)):
        #print('Layer {}'.format(layer_idx))
        for param in layer.get_params(trainable=True):
            #print('    param: {}'.format(param))
            param_layer_lookup[param] = (layer_idx, str(param))
    # Abs-Sum all gradients per parameter, and store a lookup
    #   that tells us the layer and parameter name for that grad
    all_grads = []
    grads_to_layer = [-1 for _ in range(len(steps.keys()))]
    # Change updates to steps, here!
    for p_idx, (param, grad) in enumerate(steps.items()):
        all_grads.append(T.sum(T.abs_(grad)))        
        # Store a lookup if a real param - not momentum, etc.
        if param in param_layer_lookup:
            grads_to_layer[p_idx] = param_layer_lookup[param]

    all_grads = T.as_tensor_variable(all_grads)
    #grads_to_layer= T.as_tensor_variable(grads_to_layer)
    #train_fn = theano.function(fn_inputs, [loss, train_acc,prediction_thresholded], updates=updates)

    train_fn = theano.function(fn_inputs, [loss,train_acc,prediction_thresholded,all_grads,activations], updates=updates,allow_input_downcast=True)

    val_fn = theano.function(fn_inputs, [test_loss, test_acc,test_prediction_thresholded])


    return train_fn, val_fn, out_fn

#determine the output labels by selecting the largest 6 vaules in the output layer activations
#instead of thresholding
def get_val_fn_max6(inputs, target_var, network, k=6):
    test_prediction = lasagne.layers.get_output(network, deterministic=True)

    
    ind=T.argsort(test_prediction, axis=1)[:,-k:]
    ind_x = T.argmax(T.eye(ind.shape[0]), axis=1).repeat(ind.shape[1])
    ind_y = T.flatten(ind)
    test_pred_arr=T.zeros_like(test_prediction)
    test_pred_arr = T.set_subtensor(test_pred_arr[ind_x, ind_y], 1)
        
    # Complicated math to get the mean per-example error rate
    
    sum_of_targets = T.sum(target_var, axis=1)
    test_mean_mismatch_per_example = T.sum(abs(test_pred_arr-target_var), axis=1)/sum_of_targets
    test_acc = 1.-T.mean(test_mean_mismatch_per_example, dtype=theano.config.floatX)
    test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
    test_loss = test_loss.mean()

    out_fn = get_layer_output_fn(inputs, network)
    fn_inputs = inputs + [target_var]
    
    val_fn = theano.function(fn_inputs,[ind, test_loss, test_acc,test_pred_arr,test_prediction])

    return val_fn

class MatlabLipReadingIterator(object):
    def flow(self, eng, data_keys, data_dirs, batch_size, shuffle=True, blankout=0, flatten_color=True):
        # Get some constants
        num_examples = len(data_keys)
        num_batches = int(np.ceil(float(num_examples)/batch_size))
        # Shuffle the data
        if shuffle:
            np.random.shuffle(data_keys)
        b = 0
        next_keys = data_keys[b*batch_size:(b+1)*batch_size]
        # Issue asynchronous job
        future = issue_matlab_lipreading_dataprep_job(eng, next_keys, data_dirs, blankout)
        while b < num_batches:
            aud_data, vid_data, bY = get_matlab_lipreading_results(eng, future)
            # Issue async next job just before returning and computing
            b += 1
            if b < num_batches:
                next_keys = data_keys[b*batch_size:(b+1)*batch_size]
                future = issue_matlab_lipreading_dataprep_job(eng, next_keys, data_dirs, blankout)
            # Return and let computation commence
            yield aud_data, vid_data, bY

# Matlab Functions
# ----------------------------------------------------
def matlab_mat_to_np(var):
    # Reshape from matlab to numpy
    return np.array(var._data).reshape(var.size[::-1]).T

def get_matlab_lipreading_results(eng, future):
    while not future.done():
        time.sleep(0.050)
    aud_data, vid_data, bY = future.result()
    aud_data = matlab_mat_to_np(aud_data).astype('float32')
    vid_data = matlab_mat_to_np(vid_data).astype('float32')
    bY       = matlab_mat_to_np(bY).astype('float32')
    return aud_data, vid_data, bY

def issue_matlab_lipreading_dataprep_job(eng, file_keys, data_dirs, blankout):
    clean_strings = [str(curr_file) for curr_file in file_keys]
    data_dirs_clean = [str(data_dir) for data_dir in data_dirs]
    future = eng.batch_lipreading_dataprep(clean_strings, data_dirs_clean, blankout, nargout=3, async=True)
    return future
