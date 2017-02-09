Scripts for visualize cochlea, DAVIS and microphone recordings

Main script: data_prep.m
		Please modify the pathname and word name at the begining of data_prep.m before plotting
		Run the last section only for plot if the data has already been loaded

plot_cochlea_retina_DAVIS_3D.m
		A function that take loaded data and plot 2 sets of figures: 	1) cochlea with microphone
										2) cochlea only, zoom in
										3) a set of DAVIS snapshots that cover a certain time interval
plot_ret_3d.m
		A script that generate a set of DAVIS snapshots from a certain interval. 
		Change the value of 'interested_interval' to select what time interval to plot
		Change 'row' and 'col' to adjust the number of snapshots