% basefil is the base file name (string with no extension but relative path)
function plot_cochlea_retina_DAVIS_3D(sensor_data)

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

close all;
ret_x = 240; ret_y = 180;
%jaer_startlat = 3.2; jaer_stoplat = 0.8; % jAER starting latency (both sensors), and stopping latency (single sensor)
jaer_startlat=0;
jaer_stoplat=0;
beep_dur=0;
%beep_dur = 0.4; % beep duration (estimated), sec

sf=size(frames);

% Calculate time axes allTs_ret
allTs_ret=double(allTs_ret)/1e6;
frameEnds=double(frameEndsTs);
%allTsd_ret = (allTs_ret-min(allTs_ret))/1e6;
%allTsd_frames = (frameEnds-min(allTs_ret))/1e6;
cnt_frames=1;
%coch_ret_offset = max(tsright) - max(allTsd_ret) + jaer_stoplat; % Timing offset caused by jAER logging delays
%allTsd_ret = allTsd_ret + coch_ret_offset; allTsd_frames = allTsd_frames + coch_ret_offset;
max_Tsd_coc=max([max(tsleft) max(tsright) max(t_aud)]);

%tot = max([max(allTsd_ret) max(allTsd_frames) max(tsleft) max(tsright) max(t_aud)]);
%nt = floor(tot/dt);
figure(1); clf;
        subplot(2,1,1)
        plot(tsleft,laddress,'b.'); hold on;
        plot(tsright,raddress,'k.');
        xlabel('Time (sec)'); xlim([0 max_Tsd_coc]);
        title('Cochlea data');

        subplot(2,1,2)
        plot(t_aud,aud(:,1),'b.'); hold on;
%        plot(t_aud,aud(:,2)+0.5,'k.');
        xlabel('Time (sec)'); xlim([0 max_Tsd_coc]);
        title('Microphone data');

        figure(2);clf;
        %subplot(2,1,1)
        %ind0=find(floor(laddress/4)==10);
        ind0=1:length(laddress);
        mini_gap=0.1;
        for cnt=1:length(ind0)-1
            if(tsleft(ind0(cnt+1))-tsleft(ind0(cnt))>mini_gap)
                break;
            end
        end
        
        beep_delay=tsleft(ind0(cnt));
        digit_end=tsleft(ind0(end));
        ind1=find((tsleft>beep_delay)&(tsleft<digit_end));
        ind2=find((tsright>beep_delay)&(tsright<digit_end));
        plot(tsleft(ind1),laddress(ind1),'b.'); hold on;
        plot(tsright(ind2),raddress(ind2),'k.');
        
        xlabel('Time (sec)'); %xlim([min(tsleft(ind1)) max(tsleft(ind1))]);
        title('Cochlea data- no beep');

%         subplot(2,1,2)
%         plot(t_aud,aud(:,1),'b.'); hold on;
% %        plot(t_aud,aud(:,2)+0.5,'k.');
%         xlabel('Time (sec)'); xlim([0 max_Tsd_coc]);
%         title('Microphone data');
        
        plot_ret_3d;

%     
    ind=find((allTsd_frames>(i-1)*dt) & (allTsd_frames<i*dt));
    if ~isempty(ind) && cnt_frames < sf(4)+1
        subplot(2,2,3);
        imagesc(flipud(frame_int(:,:,cnt_frames)')); 
        cnt_frames=cnt_frames+1;
        xlim([0 ret_x]); ylim([0 ret_y]); axis equal; colormap gray;
        title('Retina frame data');
    end
    


end