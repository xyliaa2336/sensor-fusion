%% load data
clear all;close all; clc

%change pathname to the path where recordings are stored
pathname='D:\lipread_data\hw_grid_DVS_part2\';

fprintf('...loading data...\n')
% change word name to plot out different words
[sensor_data]=load_cochlea_DAVIS([pathname,'bin_blue_at_k_1_now\']);
%% unpack the data from 'sensor_data' for debugging
%read loaded data
tsleft=sensor_data{1}{1};
laddress=sensor_data{1}{2};
tsright=sensor_data{1}{3};
raddress=sensor_data{1}{4}; 
% %%%
% right_ind=find(tsright==0);
% tsright=tsright(right_ind+1:end);
% raddress=raddress(right_ind+1:end);

t_aud=sensor_data{2}{1};
aud=sensor_data{2}{2};
Fs=sensor_data{2}{3};

x=sensor_data{3}{1};
y=sensor_data{3}{2};
pol=sensor_data{3}{3};
allTs_ret=sensor_data{3}{4};

frameEndsTs=sensor_data{4}{1};
frames=sensor_data{4}{2};
frame_int=sensor_data{4}{3};
%% plot figures. Run this section only if data has already been loaded
fprintf('...plotting...\n')
plot_cochlea_retina_DAVIS_3D(sensor_data);