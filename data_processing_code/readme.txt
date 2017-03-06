Procedure of converting raw event data to hdf5 format:

Matlab part:
1)	Go to ¡°dataprocessing_matlab¡± folder
2)	Run ¡°cochLP_create_mat.m¡± and ¡°DVS_create_mat.m¡± to load the aedat recordinds and save the raw recordings in mat format
3)	Use a version of ¡°CochLP2vector.m¡± to make spikegrams. 
Available versions: bin by constant time; bin by constant number of events from EITHER on OR off events; bin by constant number of events from on AND off events
4)	Use a version of ¡°DVS2frame.m¡± to make binned DVS frames.
Available versions: bin by constant time, on or off events; bin by constant time, on - off events.
5)	To compute correlated DVS spikes, use ¡°DVS2frame_corr.m¡±. Please note that this may take much longer than other binning methods. 

Remark: The normalization is done in ¡°CochLP2vector.m¡± or ¡°DVS2frame.m¡±. The script ¡°Coch_norm.m¡± can be used to normalize data if they are already processed.

Python Part:
Please follow the procedure in ¡°Creat_hdf5_from_mat.ipynb¡± to covert the mat files created in last steps into hdf5 format.

For event data, all pre-processing are done in matlab scripts. 
