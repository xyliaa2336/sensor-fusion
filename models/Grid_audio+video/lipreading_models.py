import lasagne
from lipreading_utils import vocab_size

def get_dan_original_rnn(aud_in, vid_in):
    # INPUTS
    #   (batch size, max sequence length, number of features)
    l_aud_in = lasagne.layers.InputLayer(shape=(None, None, 39), input_var=aud_in)
    #   (batch size, max sequence length, channels, rows, cols)
    l_vid_in = lasagne.layers.InputLayer(shape=(None, None, 1, 48, 48), input_var=vid_in)
    batch_size, vid_time_length = vid_in.shape[0], vid_in.shape[1]    
    # Flatten together batches and sequences
    #    First reverse it so that the flatten dims are at the end
    dimshuffle_vid_in = lasagne.layers.DimshuffleLayer(l_vid_in, (4,3,2,1,0))
    #    Now flatten it: (48 x 48 x 1 x seq_len x batch_size) => (48 x 48 x 1 x seq_len*batch_size)
    flatten_vid = lasagne.layers.FlattenLayer(dimshuffle_vid_in, outdim=4)
    #    Now unreverse it: (seq_len*batch_size x 1 x 48 x 48)
    reshape_vid_in = lasagne.layers.DimshuffleLayer(flatten_vid, (3,2,1,0))
    
    # VIDEO CONVNET BRANCH
    # Another convolution with 8 5x5 kernels, and another 2x2 pooling:
    vid_conv_1 = lasagne.layers.Conv2DLayer(reshape_vid_in, 
                                            num_filters=8, filter_size=(5, 5),
                                            nonlinearity=lasagne.nonlinearities.leaky_rectify)
    vid_maxpool_1 = lasagne.layers.MaxPool2DLayer(vid_conv_1, pool_size=(2, 2))
    vid_conv_2 = lasagne.layers.Conv2DLayer(vid_maxpool_1, 
                                            num_filters=8, filter_size=(5, 5),
                                            nonlinearity=lasagne.nonlinearities.leaky_rectify)
    vid_maxpool_2 = lasagne.layers.MaxPool2DLayer(vid_conv_2, pool_size=(2, 2))    
    vid_conv_3 = lasagne.layers.Conv2DLayer(vid_maxpool_2, 
                                            num_filters=8, filter_size=(5, 5),
                                            nonlinearity=lasagne.nonlinearities.leaky_rectify)
    vid_maxpool_3 = lasagne.layers.MaxPool2DLayer(vid_conv_3, pool_size=(2, 2))      
    # Now we want to break the timesteps back out
    #   First reverse:
    vid_final_dimshuffle = lasagne.layers.DimshuffleLayer(vid_maxpool_3, (3,2,1,0))
    #   Then pull out the timesteps:
    vid_final_reshape = lasagne.layers.ReshapeLayer(vid_final_dimshuffle, ([0], [1], [2], vid_time_length, batch_size))
    #   Then unreverse:
    vid_final_reorder = lasagne.layers.DimshuffleLayer(vid_final_reshape, (4,3,2,1,0))
    #   Then flatten the image dimensions to a vector:
    vid_final = lasagne.layers.FlattenLayer(vid_final_reorder, outdim=3)
    
    # VIDEO RNN
    vid_lstm = lasagne.layers.GRULayer(vid_final, num_units=80)
    
    # AUDIO RNN
    aud_lstm = lasagne.layers.GRULayer(l_aud_in, num_units=150)
    
    # MERGE BRANCHES
    merged = lasagne.layers.ConcatLayer([vid_lstm, aud_lstm], axis=2)

    # POST-MERGE RNNS
    full_lstm1 = lasagne.layers.GRULayer(merged, num_units=240)
    full_lstm2 = lasagne.layers.GRULayer(full_lstm1, num_units=250)
    
    # Final classification
    l_slice = lasagne.layers.SliceLayer(full_lstm2, -1, axis=1)
    l_out = lasagne.layers.DenseLayer(l_slice, num_units=vocab_size, nonlinearity=None)
    
    return l_out

def get_dan_original_audio(aud_in):
    # INPUTS
    #   (batch size, max sequence length, number of features)
    l_aud_in = lasagne.layers.InputLayer(shape=(None, None, 39), input_var=aud_in)
    
    # AUDIO RNN
    aud_lstm = lasagne.layers.GRULayer(l_aud_in, num_units=150)
    
    # POST-MERGE RNNS
    full_lstm1 = lasagne.layers.GRULayer(aud_lstm, num_units=240)
    full_lstm2 = lasagne.layers.GRULayer(full_lstm1, num_units=250)
    
    # Final classification
    l_slice = lasagne.layers.SliceLayer(full_lstm2, -1, axis=1)
    l_out = lasagne.layers.DenseLayer(l_slice, num_units=vocab_size, nonlinearity=None)
    
    return l_out

def get_dan_original_video( vid_in):
    # INPUTS
    
    #   (batch size, max sequence length, channels, rows, cols)
    l_vid_in = lasagne.layers.InputLayer(shape=(None, None, 1, 96, 96), input_var=vid_in)
    batch_size, vid_time_length = vid_in.shape[0], vid_in.shape[1]    
    # Flatten together batches and sequences
    #    First reverse it so that the flatten dims are at the end
    dimshuffle_vid_in = lasagne.layers.DimshuffleLayer(l_vid_in, (4,3,2,1,0))
    #    Now flatten it: (48 x 48 x 1 x seq_len x batch_size) => (48 x 48 x 1 x seq_len*batch_size)
    flatten_vid = lasagne.layers.FlattenLayer(dimshuffle_vid_in, outdim=4)
    #    Now unreverse it: (seq_len*batch_size x 1 x 48 x 48)
    reshape_vid_in = lasagne.layers.DimshuffleLayer(flatten_vid, (3,2,1,0))
    
    # VIDEO CONVNET BRANCH
    # Another convolution with 8 5x5 kernels, and another 2x2 pooling:
    vid_conv_1 = lasagne.layers.Conv2DLayer(reshape_vid_in, 
                                            num_filters=8, filter_size=(5, 5),
                                            nonlinearity=lasagne.nonlinearities.rectify)
    vid_maxpool_1 = lasagne.layers.MaxPool2DLayer(vid_conv_1, pool_size=(2, 2))
    vid_conv_2 = lasagne.layers.Conv2DLayer(vid_maxpool_1, 
                                            num_filters=8, filter_size=(5, 5),
                                            nonlinearity=lasagne.nonlinearities.rectify)
    vid_maxpool_2 = lasagne.layers.MaxPool2DLayer(vid_conv_2, pool_size=(2, 2))    
    vid_conv_3 = lasagne.layers.Conv2DLayer(vid_maxpool_2, 
                                            num_filters=8, filter_size=(5, 5),
                                            nonlinearity=lasagne.nonlinearities.rectify)
    vid_maxpool_3 = lasagne.layers.MaxPool2DLayer(vid_conv_3, pool_size=(2, 2))      
    # Now we want to break the timesteps back out
    #   First reverse:
    vid_final_dimshuffle = lasagne.layers.DimshuffleLayer(vid_maxpool_3, (3,2,1,0))
    #   Then pull out the timesteps:
    vid_final_reshape = lasagne.layers.ReshapeLayer(vid_final_dimshuffle, ([0], [1], [2], vid_time_length, batch_size))
    #   Then unreverse:
    vid_final_reorder = lasagne.layers.DimshuffleLayer(vid_final_reshape, (4,3,2,1,0))
    #   Then flatten the image dimensions to a vector:
    vid_final = lasagne.layers.FlattenLayer(vid_final_reorder, outdim=3)
    
    # VIDEO RNN
    vid_lstm = lasagne.layers.GRULayer(vid_final, num_units=80)
    
    

    # POST-MERGE RNNS
    full_lstm1 = lasagne.layers.GRULayer(vid_lstm, num_units=240)
    full_lstm2 = lasagne.layers.GRULayer(full_lstm1, num_units=250)
    
    # Final classification
    l_slice = lasagne.layers.SliceLayer(full_lstm2, -1, axis=1)
    l_out = lasagne.layers.DenseLayer(l_slice, num_units=vocab_size, nonlinearity=None)
    
    return l_out

def get_dan_original_video_48( vid_in):
    # INPUTS
    
    #   (batch size, max sequence length, channels, rows, cols)
    l_vid_in = lasagne.layers.InputLayer(shape=(None, None, 1, 48, 48), input_var=vid_in)
    batch_size, vid_time_length = vid_in.shape[0], vid_in.shape[1]    
    # Flatten together batches and sequences
    #    First reverse it so that the flatten dims are at the end
    dimshuffle_vid_in = lasagne.layers.DimshuffleLayer(l_vid_in, (4,3,2,1,0))
    #    Now flatten it: (48 x 48 x 1 x seq_len x batch_size) => (48 x 48 x 1 x seq_len*batch_size)
    flatten_vid = lasagne.layers.FlattenLayer(dimshuffle_vid_in, outdim=4)
    #    Now unreverse it: (seq_len*batch_size x 1 x 48 x 48)
    reshape_vid_in = lasagne.layers.DimshuffleLayer(flatten_vid, (3,2,1,0))
    
    # VIDEO CONVNET BRANCH
    # Another convolution with 8 5x5 kernels, and another 2x2 pooling:
    vid_conv_1 = lasagne.layers.Conv2DLayer(reshape_vid_in, 
                                            num_filters=8, filter_size=(5, 5),
                                            nonlinearity=lasagne.nonlinearities.rectify)
    vid_maxpool_1 = lasagne.layers.MaxPool2DLayer(vid_conv_1, pool_size=(2, 2))
    vid_conv_2 = lasagne.layers.Conv2DLayer(vid_maxpool_1, 
                                            num_filters=8, filter_size=(5, 5),
                                            nonlinearity=lasagne.nonlinearities.rectify)
    vid_maxpool_2 = lasagne.layers.MaxPool2DLayer(vid_conv_2, pool_size=(2, 2))    
    vid_conv_3 = lasagne.layers.Conv2DLayer(vid_maxpool_2, 
                                            num_filters=8, filter_size=(5, 5),
                                            nonlinearity=lasagne.nonlinearities.rectify)
    vid_maxpool_3 = lasagne.layers.MaxPool2DLayer(vid_conv_3, pool_size=(2, 2))      
    # Now we want to break the timesteps back out
    #   First reverse:
    vid_final_dimshuffle = lasagne.layers.DimshuffleLayer(vid_maxpool_3, (3,2,1,0))
    #   Then pull out the timesteps:
    vid_final_reshape = lasagne.layers.ReshapeLayer(vid_final_dimshuffle, ([0], [1], [2], vid_time_length, batch_size))
    #   Then unreverse:
    vid_final_reorder = lasagne.layers.DimshuffleLayer(vid_final_reshape, (4,3,2,1,0))
    #   Then flatten the image dimensions to a vector:
    vid_final = lasagne.layers.FlattenLayer(vid_final_reorder, outdim=3)
    
    # VIDEO RNN
    vid_lstm = lasagne.layers.GRULayer(vid_final, num_units=80)
    
    

    # POST-MERGE RNNS
    full_lstm1 = lasagne.layers.GRULayer(vid_lstm, num_units=240)
    full_lstm2 = lasagne.layers.GRULayer(full_lstm1, num_units=250)
    
    # Final classification
    l_slice = lasagne.layers.SliceLayer(full_lstm2, -1, axis=1)
    l_out = lasagne.layers.DenseLayer(l_slice, num_units=vocab_size, nonlinearity=None)
    
    return l_out
    
def get_network(aud_in, vid_in):
    # INPUTS
    #   (batch size, max sequence length, number of features)
    l_aud_in = lasagne.layers.InputLayer(shape=(None, None, 64), input_var=aud_in)
    #   (batch size, max sequence length, channels, rows, cols)
    l_vid_in = lasagne.layers.InputLayer(shape=(None, None, 1, 48, 48), input_var=vid_in)
    batch_size, vid_time_length = vid_in.shape[0], vid_in.shape[1]    
    # Flatten together batches and sequences
    #    First reverse it so that the flatten dims are at the end
    dimshuffle_vid_in = lasagne.layers.DimshuffleLayer(l_vid_in, (4,3,2,1,0))
    #    Now flatten it: (48 x 48 x 1 x seq_len x batch_size) => (48 x 48 x 1 x seq_len*batch_size)
    flatten_vid = lasagne.layers.FlattenLayer(dimshuffle_vid_in, outdim=4)
    #    Now unreverse it: (seq_len*batch_size x 1 x 48 x 48)
    reshape_vid_in = lasagne.layers.DimshuffleLayer(flatten_vid, (3,2,1,0))
    
    # VIDEO CONVNET BRANCH
    # Another convolution with 8 5x5 kernels, and another 2x2 pooling:
    vid_conv_1 = lasagne.layers.Conv2DLayer(reshape_vid_in, 
                                            num_filters=8, filter_size=(5, 5),
                                            nonlinearity=lasagne.nonlinearities.rectify)
    vid_maxpool_1 = lasagne.layers.MaxPool2DLayer(vid_conv_1, pool_size=(2, 2))
    vid_conv_2 = lasagne.layers.Conv2DLayer(vid_maxpool_1, 
                                            num_filters=8, filter_size=(5, 5),
                                            nonlinearity=lasagne.nonlinearities.rectify)
    vid_maxpool_2 = lasagne.layers.MaxPool2DLayer(vid_conv_2, pool_size=(2, 2))    
    vid_conv_3 = lasagne.layers.Conv2DLayer(vid_maxpool_2, 
                                            num_filters=8, filter_size=(5, 5),
                                            nonlinearity=lasagne.nonlinearities.rectify)
    vid_maxpool_3 = lasagne.layers.MaxPool2DLayer(vid_conv_3, pool_size=(2, 2))      
    # Now we want to break the timesteps back out
    #   First reverse:
    vid_final_dimshuffle = lasagne.layers.DimshuffleLayer(vid_maxpool_3, (3,2,1,0))
    #   Then pull out the timesteps:
    vid_final_reshape = lasagne.layers.ReshapeLayer(vid_final_dimshuffle, ([0], [1], [2], vid_time_length, batch_size))
    #   Then unreverse:
    vid_final_reorder = lasagne.layers.DimshuffleLayer(vid_final_reshape, (4,3,2,1,0))
    #   Then flatten the image dimensions to a vector:
    vid_final = lasagne.layers.FlattenLayer(vid_final_reorder, outdim=3)
    
    # VIDEO RNN
    vid_lstm = lasagne.layers.GRULayer(vid_final, num_units=80)
    
    vid_dropout=lasagne.layers.dropout(vid_lstm, p=0.5)
    
    # AUDIO RNN
    aud_lstm = lasagne.layers.GRULayer(l_aud_in, num_units=100)
    aud_dropout=lasagne.layers.dropout(aud_lstm, p=0.5)
    
    # MERGE BRANCHES
    merged = lasagne.layers.ConcatLayer([vid_dropout, aud_dropout], axis=2)

    # POST-MERGE RNNS
    full_lstm1 = lasagne.layers.GRULayer(merged, num_units=100)
    full_lstm2 = lasagne.layers.GRULayer(full_lstm1, num_units=100)
    
    # Final classification
    l_slice = lasagne.layers.SliceLayer(full_lstm2, -1, axis=1)
    merge_dropout=lasagne.layers.dropout(l_slice, p=0.5)
    l_out = lasagne.layers.DenseLayer(merge_dropout, num_units=vocab_size, nonlinearity=None)
    
    return l_out


def get_network_12classes(aud_in, vid_in):
    # INPUTS
    #   (batch size, max sequence length, number of features)
    l_aud_in = lasagne.layers.InputLayer(shape=(None, None, 39), input_var=aud_in)
    #   (batch size, max sequence length, channels, rows, cols)
    l_vid_in = lasagne.layers.InputLayer(shape=(None, None, 1, 48, 48), input_var=vid_in)
    batch_size, vid_time_length = vid_in.shape[0], vid_in.shape[1]    
    # Flatten together batches and sequences
    #    First reverse it so that the flatten dims are at the end
    dimshuffle_vid_in = lasagne.layers.DimshuffleLayer(l_vid_in, (4,3,2,1,0))
    #    Now flatten it: (48 x 48 x 1 x seq_len x batch_size) => (48 x 48 x 1 x seq_len*batch_size)
    flatten_vid = lasagne.layers.FlattenLayer(dimshuffle_vid_in, outdim=4)
    #    Now unreverse it: (seq_len*batch_size x 1 x 48 x 48)
    reshape_vid_in = lasagne.layers.DimshuffleLayer(flatten_vid, (3,2,1,0))
    
    # VIDEO CONVNET BRANCH
    # Another convolution with 8 5x5 kernels, and another 2x2 pooling:
    vid_conv_1 = lasagne.layers.Conv2DLayer(reshape_vid_in, 
                                            num_filters=8, filter_size=(5, 5),
                                            nonlinearity=lasagne.nonlinearities.rectify)
    vid_maxpool_1 = lasagne.layers.MaxPool2DLayer(vid_conv_1, pool_size=(2, 2))
    vid_conv_2 = lasagne.layers.Conv2DLayer(vid_maxpool_1, 
                                            num_filters=8, filter_size=(5, 5),
                                            nonlinearity=lasagne.nonlinearities.rectify)
    vid_maxpool_2 = lasagne.layers.MaxPool2DLayer(vid_conv_2, pool_size=(2, 2))    
    vid_conv_3 = lasagne.layers.Conv2DLayer(vid_maxpool_2, 
                                            num_filters=8, filter_size=(5, 5),
                                            nonlinearity=lasagne.nonlinearities.rectify)
    vid_maxpool_3 = lasagne.layers.MaxPool2DLayer(vid_conv_3, pool_size=(2, 2))      
    # Now we want to break the timesteps back out
    #   First reverse:
    vid_final_dimshuffle = lasagne.layers.DimshuffleLayer(vid_maxpool_3, (3,2,1,0))
    #   Then pull out the timesteps:
    vid_final_reshape = lasagne.layers.ReshapeLayer(vid_final_dimshuffle, ([0], [1], [2], vid_time_length, batch_size))
    #   Then unreverse:
    vid_final_reorder = lasagne.layers.DimshuffleLayer(vid_final_reshape, (4,3,2,1,0))
    #   Then flatten the image dimensions to a vector:
    vid_final = lasagne.layers.FlattenLayer(vid_final_reorder, outdim=3)
    
    # VIDEO RNN
    vid_lstm = lasagne.layers.GRULayer(vid_final, num_units=40)
    
    vid_dropout=lasagne.layers.dropout(vid_lstm, p=0.5)
    
    # AUDIO RNN
    aud_lstm = lasagne.layers.GRULayer(l_aud_in, num_units=50)
    aud_dropout=lasagne.layers.dropout(aud_lstm, p=0.5)
    
    # MERGE BRANCHES
    merged = lasagne.layers.ConcatLayer([vid_dropout, aud_dropout], axis=2)

    # POST-MERGE RNNS
    full_lstm1 = lasagne.layers.GRULayer(merged, num_units=100)
    full_lstm2 = lasagne.layers.GRULayer(full_lstm1, num_units=100)
    
    # Final classification
    l_slice = lasagne.layers.SliceLayer(full_lstm2, -1, axis=1)
    merge_dropout=lasagne.layers.dropout(l_slice, p=0.5)
    l_out = lasagne.layers.DenseLayer(merge_dropout, num_units=vocab_size+1, nonlinearity=None)
    
    return l_out

def get_network_12classes_events(aud_in, vid_in):

    # INPUTS
    #   (batch size, max sequence length, number of features)
    l_aud_in = lasagne.layers.InputLayer(shape=(None, None, 64), input_var=aud_in)
    #   (batch size, max sequence length, channels, rows, cols)
    l_vid_in = lasagne.layers.InputLayer(shape=(None, None, 1, 180, 180), input_var=vid_in)
    batch_size, vid_time_length = vid_in.shape[0], vid_in.shape[1]    
    # Flatten together batches and sequences
    #    First reverse it so that the flatten dims are at the end
    dimshuffle_vid_in = lasagne.layers.DimshuffleLayer(l_vid_in, (4,3,2,1,0))
    #    Now flatten it: (48 x 48 x 1 x seq_len x batch_size) => (48 x 48 x 1 x seq_len*batch_size)
    flatten_vid = lasagne.layers.FlattenLayer(dimshuffle_vid_in, outdim=4)
    #    Now unreverse it: (seq_len*batch_size x 1 x 48 x 48)
    reshape_vid_in = lasagne.layers.DimshuffleLayer(flatten_vid, (3,2,1,0))
    
    # VIDEO CONVNET BRANCH
    # Another convolution with 8 5x5 kernels, and another 2x2 pooling:
    vid_conv_1 = lasagne.layers.Conv2DLayer(reshape_vid_in, 
                                            num_filters=8, filter_size=(5, 5),
                                            nonlinearity=lasagne.nonlinearities.rectify)
    vid_maxpool_1 = lasagne.layers.MaxPool2DLayer(vid_conv_1, pool_size=(2, 2))
    vid_conv_2 = lasagne.layers.Conv2DLayer(vid_maxpool_1, 
                                            num_filters=8, filter_size=(5, 5),
                                            nonlinearity=lasagne.nonlinearities.rectify)
    vid_maxpool_2 = lasagne.layers.MaxPool2DLayer(vid_conv_2, pool_size=(2, 2))    
    vid_conv_3 = lasagne.layers.Conv2DLayer(vid_maxpool_2, 
                                            num_filters=8, filter_size=(5, 5),
                                            nonlinearity=lasagne.nonlinearities.rectify)
    vid_maxpool_3 = lasagne.layers.MaxPool2DLayer(vid_conv_3, pool_size=(2, 2))      
    # Now we want to break the timesteps back out
    #   First reverse:
    vid_final_dimshuffle = lasagne.layers.DimshuffleLayer(vid_maxpool_3, (3,2,1,0))
    #   Then pull out the timesteps:
    vid_final_reshape = lasagne.layers.ReshapeLayer(vid_final_dimshuffle, ([0], [1], [2], vid_time_length, batch_size))
    #   Then unreverse:
    vid_final_reorder = lasagne.layers.DimshuffleLayer(vid_final_reshape, (4,3,2,1,0))
    #   Then flatten the image dimensions to a vector:
    vid_final = lasagne.layers.FlattenLayer(vid_final_reorder, outdim=3)
    
    # VIDEO RNN
    vid_lstm = lasagne.layers.GRULayer(vid_final, num_units=40)
    
    vid_dropout=lasagne.layers.dropout(vid_lstm, p=0.5)
    
    # AUDIO RNN
    aud_lstm = lasagne.layers.GRULayer(l_aud_in, num_units=50)
    aud_dropout=lasagne.layers.dropout(aud_lstm, p=0.5)
    
    # MERGE BRANCHES
    merged = lasagne.layers.ConcatLayer([vid_dropout, aud_dropout], axis=2)

    # POST-MERGE RNNS
    full_lstm1 = lasagne.layers.GRULayer(merged, num_units=100)
    full_lstm2 = lasagne.layers.GRULayer(full_lstm1, num_units=100)
    
    # Final classification
    l_slice = lasagne.layers.SliceLayer(full_lstm2, -1, axis=1)
    merge_dropout=lasagne.layers.dropout(l_slice, p=0.5)
    l_out = lasagne.layers.DenseLayer(merge_dropout, num_units=vocab_size+1, nonlinearity=None)
    
    return l_out

def get_video_coch_net(aud_in, vid_in):

    # INPUTS
    #   (batch size, max sequence length, number of features)
    l_aud_in = lasagne.layers.InputLayer(shape=(None, None, 64), input_var=aud_in)
    #   (batch size, max sequence length, channels, rows, cols)
    l_vid_in = lasagne.layers.InputLayer(shape=(None, None, 1, 48, 48), input_var=vid_in)
    batch_size, vid_time_length = vid_in.shape[0], vid_in.shape[1]    
    # Flatten together batches and sequences
    #    First reverse it so that the flatten dims are at the end
    dimshuffle_vid_in = lasagne.layers.DimshuffleLayer(l_vid_in, (4,3,2,1,0))
    #    Now flatten it: (48 x 48 x 1 x seq_len x batch_size) => (48 x 48 x 1 x seq_len*batch_size)
    flatten_vid = lasagne.layers.FlattenLayer(dimshuffle_vid_in, outdim=4)
    #    Now unreverse it: (seq_len*batch_size x 1 x 48 x 48)
    reshape_vid_in = lasagne.layers.DimshuffleLayer(flatten_vid, (3,2,1,0))
    
    # VIDEO CONVNET BRANCH
    # Another convolution with 8 5x5 kernels, and another 2x2 pooling:
    vid_conv_1 = lasagne.layers.Conv2DLayer(reshape_vid_in, 
                                            num_filters=8, filter_size=(5, 5),
                                            nonlinearity=lasagne.nonlinearities.leaky_rectify)
    vid_maxpool_1 = lasagne.layers.MaxPool2DLayer(vid_conv_1, pool_size=(2, 2))
    vid_conv_2 = lasagne.layers.Conv2DLayer(vid_maxpool_1, 
                                            num_filters=8, filter_size=(5, 5),
                                            nonlinearity=lasagne.nonlinearities.leaky_rectify)
    vid_maxpool_2 = lasagne.layers.MaxPool2DLayer(vid_conv_2, pool_size=(2, 2))    
    vid_conv_3 = lasagne.layers.Conv2DLayer(vid_maxpool_2, 
                                            num_filters=8, filter_size=(5, 5),
                                            nonlinearity=lasagne.nonlinearities.leaky_rectify)
    vid_maxpool_3 = lasagne.layers.MaxPool2DLayer(vid_conv_3, pool_size=(2, 2))      
    # Now we want to break the timesteps back out
    #   First reverse:
    vid_final_dimshuffle = lasagne.layers.DimshuffleLayer(vid_maxpool_3, (3,2,1,0))
    #   Then pull out the timesteps:
    vid_final_reshape = lasagne.layers.ReshapeLayer(vid_final_dimshuffle, ([0], [1], [2], vid_time_length, batch_size))
    #   Then unreverse:
    vid_final_reorder = lasagne.layers.DimshuffleLayer(vid_final_reshape, (4,3,2,1,0))
    #   Then flatten the image dimensions to a vector:
    vid_final = lasagne.layers.FlattenLayer(vid_final_reorder, outdim=3)
    
    # VIDEO RNN
    vid_lstm = lasagne.layers.GRULayer(vid_final, num_units=80)
    
    # AUDIO RNN
    aud_lstm = lasagne.layers.GRULayer(l_aud_in, num_units=150)
    
    # MERGE BRANCHES
    merged = lasagne.layers.ConcatLayer([vid_lstm, aud_lstm], axis=2)

    # POST-MERGE RNNS
    full_lstm1 = lasagne.layers.GRULayer(merged, num_units=240)
    full_lstm2 = lasagne.layers.GRULayer(full_lstm1, num_units=250)
    
    # Final classification
    l_slice = lasagne.layers.SliceLayer(full_lstm2, -1, axis=1)
    #merge_dropout=lasagne.layers.dropout(l_slice, p=0.5)
    l_out = lasagne.layers.DenseLayer(l_slice, num_units=vocab_size, nonlinearity=None)
    
    return l_out