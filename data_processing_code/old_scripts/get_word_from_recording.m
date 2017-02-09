%%% This function load cochlea + DAVIS + audio data, synchronize the 3
%%% recordings and extract the word segment
function get_word_from_recording(basefil,word,des_dir)
    debug=0;
    % load data
    [sensor_data]=load_cochlea_DAVIS(basefil);
    %read loaded data
    tslefton=sensor_data{1}{1};
    addrlon=sensor_data{1}{2};
    tsleftoff=sensor_data{1}{3};
    addrloff=sensor_data{1}{4}; 
    tsrighton=sensor_data{1}{5};
    addrron=sensor_data{1}{6};
    tsrightoff=sensor_data{1}{7};
    addrroff=sensor_data{1}{8}; 
    allTsADC0=sensor_data{1}{9};
    renorm_valueADC0=sensor_data{1}{10};
    allTsADC1=sensor_data{1}{11};
    renorm_valueADC1=sensor_data{1}{12}; 
    %in case time stamp is reset after recording started
    start_indl=find(diff(tslefton)<0);
    start_indr=find(diff(tsrighton)<0); 
    if(~isempty(start_indl))
        tslefton=tslefton(start_indl+1:end);
        addrlon=addrlon(start_indl+1:end);
    end
    if(~isempty(start_indr))
        tsrighton=tsrighton(start_indr+1:end);
        addrron=addrron(start_indr+1:end);
    end
    
    tsleft=sensor_data{5}{1};
    laddress=sensor_data{5}{2};
    tsright=sensor_data{5}{3};
    raddress=sensor_data{5}{4}; 
    
    %in case time stamp is reset after recording started
    start_indl=find(diff(tsleft)<0);
    start_indr=find(diff(tsright)<0); 
    if(~isempty(start_indl))
        tsleft=tsleft(start_indl+1:end);
        laddress=laddress(start_indl+1:end);
    end
    if(~isempty(start_indr))
        tsright=tsright(start_indr+1:end);
        raddress=raddress(start_indr+1:end);
    end
        
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

    % Calculate time axes allTs_ret
    allTs_ret=double(allTs_ret)/1e6;
    frameEnds=double(frameEndsTs);
    allTsd_frames = (frameEnds)/1e6;
    % synchronize wav and coclea spikes by the beep
     
     
    sampperiod = median(diff(allTsADC0));
    avg_eng_filter=ones(1,floor(0.05/sampperiod));
    aud_eng=conv(renorm_valueADC0.^2,avg_eng_filter);
    mic_threshold_coch=mean(aud_eng(1:5000))*3
    ind_coch=(aud_eng>mic_threshold_coch);
    figure;
    plot(ind_coch);ylim([-1,2])
    beep_falling_edge=find(diff(ind_coch)==-1);
    word_rising_edge=find(diff(ind_coch)==1) %second one
    cnt=find(diff(ind_coch(word_rising_edge(2):end))==1,1)+word_rising_edge(2);
    start_coch=allTsADC0(cnt);
    
    
    beep_delay=tslefton(find(tslefton>start_coch,1));
    digit_end=find(ind_coch);
    if digit_end(end)<length(allTsADC0)
        digit_end=allTsADC0(digit_end(end))
    else
        digit_end=allTsADC0(end);
    end
    
    ind1=find((tslefton>beep_delay)&(tslefton<digit_end));
    ind2=find((tsleftoff>beep_delay)&(tsleftoff<digit_end));
    
    %debug
    if(debug==1)
        ind1=[1:length(tslefton)];
        ind2=[1:length(tsleftoff)];
    end
    % use left ear's recording
    start_coch_ind=ind1(1);
    stop_coch_ind=ind1(end);
    start_coch=tslefton(start_coch_ind)
    stop_coch=tslefton(stop_coch_ind)
    
    word_length=stop_coch-start_coch
    
    
    mic_threshold=0.5+mean(aud(1:1000))
    avg_eng_filter=ones(1,floor(0.05*Fs));
    aud_eng=conv(aud.^2,avg_eng_filter);
    ind3=(aud_eng>mic_threshold);
    figure;
    plot(ind3);ylim([-1,2])
    beep_falling_edge=find(diff(ind3)==-1);
    word_rising_edge=find(diff(ind3)==1) %second one
    start_word=t_aud(find(diff(ind3(word_rising_edge(2):end))==1,1)+word_rising_edge(2))


    %truncate audio so that the length equals to cochlea word length
    stop_wav=start_word+word_length
    stop_word_ind=find(t_aud>stop_wav);
    if(isempty(stop_word_ind))
        stop_word_ind=length(t_aud);
    end
    stop_word_ind=stop_word_ind(1)
    start_word_ind=find(t_aud==start_word) 
    %debug
    if(debug==1)
        start_word_ind=1;
        stop_word_ind=length(t_aud);
    end
    % synchronize cochlea and DAVIS time stamp ?? add 0.3 sec to both
    % starting and ending, as lip movements usually take longer than sound
    start_DAVIS=start_coch-0.3
    stop_DAVIS=stop_coch+0.3
    DAVIS_ind=find(allTs_ret>=start_DAVIS & allTs_ret<=stop_DAVIS);
    frame_ind=find(allTsd_frames>=start_DAVIS & allTsd_frames<=stop_DAVIS);
    
    % plot the word segment out for debugging
%     figure;clf;
%     subplot(2,1,1)
%     plot(tslefton(start_coch_ind:stop_coch_ind),addrlon(start_coch_ind:stop_coch_ind),'b.');
%     % hold on;plot(tsright(start_coch_ind:stop_coch_ind),raddress(start_coch_ind:stop_coch_ind),'k.');
%     xlabel('Time (sec)'); xlim([tslefton(start_coch_ind) tslefton(stop_coch_ind)]);
%     title('cochlea events')
%     subplot(2,1,2)
%     plot(t_aud(start_word_ind:stop_word_ind),aud(start_word_ind:stop_word_ind),'b.');
%     xlabel('Time (sec)'); %xlim([start_word stop_wav]);
%     title('Microphone data');
%     
    
%     check_DAVIS([start_DAVIS,stop_DAVIS],allTs_ret,x,y,allTsd_frames,frame_int);

    % save word segment: 1)wav 2)frames 3)cochleaLP events 4)DVS events 5)
    % cochleaAMS events
    tslefton=tslefton(start_coch_ind:stop_coch_ind);
    addrlon=addrlon(start_coch_ind:stop_coch_ind);
    tsleftoff=tsleftoff(ind2(1):ind2(end));
   addrloff=addrloff(ind2(1):ind2(end));
    tsrighton=(start_coch_ind:stop_coch_ind);
%    raddress=raddress(start_coch_ind:stop_coch_ind);
    %des_dir='C:\Users\YY\polybox\thesis\recording_files\cochlp_6_23_sample\';
    save([des_dir,'\',word '_cochLP.mat',], 'tslefton','addrlon', 'tsleftoff','addrloff');
    
    cochleaAMS_ind1=find(tsleft>=start_coch & tsleft<=stop_coch);
    cochleaAMS_ind2=find(tsright>=start_coch & tsright<=stop_coch);
    tsleft=tsleft(cochleaAMS_ind1);laddress=laddress(cochleaAMS_ind1);
    tsright=tsright(cochleaAMS_ind2);raddress=raddress(cochleaAMS_ind2);
    save([des_dir,'\',word '_cochAMS1C.mat',], 'tsleft','laddress', 'tsright','raddress');
    
    x=x(DAVIS_ind);y=y(DAVIS_ind);allTs_ret=allTs_ret(DAVIS_ind);
    indf=find(allTsd_frames<stop_DAVIS & allTsd_frames>start_DAVIS);
    allTsd_frames=allTsd_frames(indf);
    frame_seg=frame_int(:,:,indf);
    frame_final=zeros(size(frame_seg,2),size(frame_seg,1),length(indf));
    for i=1:length(indf)
        frame_final(:,:,i)=flipud(frame_seg(:,:,i)');
    end
    save([des_dir,'\',word,'_video.mat',], 'frame_final','allTsd_frames');
    
    save([des_dir,'\',word '_ret.mat',], 'x','y','allTs_ret');
    
    t_aud=t_aud(start_word_ind:stop_word_ind);aud=aud(start_word_ind:stop_word_ind);
    save([des_dir,'\',word,'_audio.mat',], 't_aud','aud','Fs');
    
    %save cochlp adc data to a wav file
     sampperiod = median(diff(allTsADC0));
    renorm_valueADC = [renorm_valueADC0 / max(abs(renorm_valueADC0)), renorm_valueADC1 / max(abs(renorm_valueADC1))];
    audiowrite([des_dir,'\',word,'_CochADC.wav',], renorm_valueADC, int32(1/sampperiod));
    
end
