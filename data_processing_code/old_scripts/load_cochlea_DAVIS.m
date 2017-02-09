function [sensor_data]=load_cochlea_DAVIS(basefil)
ret_x = 240; ret_y = 180;
%jaer_startlat = 3.2; jaer_stoplat = 0.8; % jAER starting latency (both sensors), and stopping latency (single sensor)
jaer_startlat = 0; jaer_stoplat = 0;
dt = 0.05; % Lumping window (sec)

disp('Loading cochleaLP data file...');
% [allAddr,allTs]=loadaerdat([basefil '_coch.aedat']);% Select cochlea data file
filename=ls([basefil 'CochleaLP*'])
[allAddr,allTs]=loadaerdat([basefil filename]);% Select cochlea data file
%[tsleft,laddress,tsright,raddress]=plotloadaecochams1c_new(allTs, allAddr);
[ tslefton, addrlon,tsleftoff,addrloff, tsrighton, addrron,tsrightoff,addrroff,allTsADC0,renorm_valueADC0, allTsADC1,renorm_valueADC1]=plotloadaeADCcochLP(allTs, allAddr);

disp('Loading cochleaams1c file...')
 filename=ls([basefil 'CochleaAMS*'])
 [allAddr,allTs]=loadaerdat([basefil filename]);% Select cochlea data file
 [tsleft,laddress,tsright,raddress]=plotloadaecochams1c_new(allTs, allAddr);
%allAddr=[];allTs=[];tsleft=[];laddress=[];tsright=[];raddress=[];



disp('Loading audio data file...');
[aud, Fs] = audioread([basefil(1:end-1) '.wav']); tmp = size(aud);
t_aud = linspace(0,tmp(1)-1,tmp(1))'/Fs; 
t_aud = t_aud + jaer_startlat;

% DVS
%[allAddr_ret,allTs_ret]=loadaerdat([basefil '_ret.aedat']); % Load retina data file;
%[x,y,pol]=extractRetina128EventsFromAddr(allAddr_ret);

% DAVIS
filename=ls([basefil 'DAVIS*'])
disp('Loading retina data file (events)...');
%[x,y,pol,allTs_ret] = getDVSeventsDavis([basefil '_ret.aedat']); % Spikes
[x,y,pol,allTs_ret] = getDVSeventsDavis([basefil filename]); % Spikes
disp('Loading retina data file (frames)...');
%[frames, frameEndsTs] = getAPSframesDavisGS([basefil '_ret.aedat']); % Frames
[frames, frameEndsTs] = getAPSframesDavisGS([basefil filename]); % Frames
sf=size(frames); frame_int = zeros(ret_x,ret_y,sf(4));
for i=1:sf(4)
    frame_int(:,:,i)=frames(3,:,:,i);
end

disp('All files loaded.');

sensor_data=cell(1,4);
%cochlea data

sensor_data{1}={ tslefton, addrlon,tsleftoff,addrloff, tsrighton, addrron,tsrightoff,addrroff,allTsADC0,renorm_valueADC0, allTsADC1,renorm_valueADC1};
%audio data
sensor_data{2}={t_aud,aud, Fs};
%DAVIS event data
sensor_data{3}={x,y,pol,allTs_ret};
%frames
sensor_data{4}={frameEndsTs,frames,frame_int};
%cochleaAMS
sensor_data{5}={tsleft,laddress,tsright,raddress};

end