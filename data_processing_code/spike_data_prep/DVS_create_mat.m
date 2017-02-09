%%
clear all;close all; clc
pathname='F:\hw_grid_DVS_part4\';

fprintf('...loading data...\n')

%%
filenames=ls([pathname]);
filenames=filenames(3:end,:);

disp('Loading retina data file (events)...');
%[x,y,pol,allTs_ret] = getDVSeventsDavis([basefil '_ret.aedat']); % Spikes
for i =1:length(filenames)
    [x,y,pol,allTs_ret] = getDVSeventsDavis([pathname,filenames(i,:)]); 
    save(['F:\hw_grid_DVS_part4_mat\',filenames(i,1:end-6),'.mat'],'x','y','pol','allTs_ret');
end
