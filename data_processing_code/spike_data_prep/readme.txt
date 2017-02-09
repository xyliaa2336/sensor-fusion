Data pre-processing scripts

Matlab part:
Function: Converting .aedat recordings to binned matrices, stored in mat format

Procedure: 1) Run cochLP_create_mat.m and DVS_create_mat.m to load the aedat recordinds and save the raw recordings in mat format
	   2) Use a version of CochLP2vector.m to make spikegrams. 
	      Available versions: bin by constant time; bin by constant number of events from EITHER on OR off events; bin by constant number of events from on AND off events
	   3) Use a version of DVS2frame.m to make binned DVS frames.
              Available versions: bin by constant time, on or off events; bin by constant time, on - off events

The processed mat files are then passed to python processing code to generate hdf5 files

Python Part:
 Run Creat_hdf5_from_mat.ipynb to create hdf5 files from processed data
