%folder that holds the recording files(DAVIS, cochLP and cochams1c)
%please comment corresponding lines in load_cochlea_DAVIS function if there
%is fewer data files
basefile='.\sample_recording\bin_blue_at_k_1_now\';

[sensor_data]=load_cochlea_DAVIS(basefile);

%read loaded data
tsleft=sensor_data{1}{1};
laddress=sensor_data{1}{2};
tsright=sensor_data{1}{3};
raddress=sensor_data{1}{4}; 

t_aud=sensor_data{2}{1};
aud=sensor_data{2}{2};
Fs=sensor_data{2}{3};

x=sensor_data{3}{1};
y=sensor_data{3}{2};
pol=sensor_data{3}{3};
allTs_ret=sensor_data{3}{4};
close all;
allTs_ret=double(allTs_ret)/1e6;
plot3(allTs_ret,x,y,'.');xlabel('t');ylabel('x');zlabel('y');

xl = get(gca,'xlim');
yl = get(gca,'ylim');
zl = get(gca,'zlim');
l = size(tsleft,1);
view([-30 20]);
%Show the image
hold on
%plot cochlea
%plot(tsleft,laddress,'r.'); 
plot3(tsleft,repmat(zl(1),l,1),laddress,'r.');
