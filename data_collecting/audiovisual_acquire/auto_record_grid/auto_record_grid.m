%% automaticly play Grid audio/video files and record with Cochlea/DAVIS
% select certain speakers to record a subset of the whole dataset 
% or set variable "dropout" to randomly drop out some of the files.
% use fixed random seed for reproducibility 

%% initialization
%open udp port, u1 for cochlea, u2 for retina
% u1=udp('localhost',8997); % make a UDP interface to localhost on port 8995 (the default port for AEViewer RemoteControl interface)
% fopen(u1);
 %u2=udp('localhost',8997); % make a UDP interface to localhost on port 8997 (the default port for AEViewer RemoteControl interface)
%fopen(u2)

%set parameters for external microphone, if any
% samp=48000
% rec_dur=4;
% recorder=recorder = audiorecorder(samp,16,1); % 16-bits

% default_jpath='C:\Users\YY\polybox\thesis\recording_files\tests\';

% drop out precentage
dropout=0;
rng(26)
rec_cnt=0;

%% recording loop
fprintf('Start recording...\n')
rootpath='D:\lipreading_raw_data\high_res\';
dest_dir='D:\lipread_data\hw_grid_DVS_part2\'; % directory to save sensor files
%dest_dir='C:\Users\YY\polybox\thesis\lipreading\audiovisual_acquire\DVS_tests';

% Create a cascade detector object.
faceDetector = vision.CascadeObjectDetector();
%% DAVIS calibration
readerobj = VideoReader([rootpath 's1' '\' 'bbaf2n.mpg']);
% Read in all video frames.
vidFrames = read(readerobj);

% Get the number of frames.
%numFrames = get(readerobj, 'NumberOfFrames');
numFrames=size(vidFrames,4);
videoFrame      = vidFrames(:,:,:,1);
bbox = step(faceDetector, videoFrame);
            box_range=[bbox(2)-40,bbox(2)+bbox(4)+40,bbox(1)-50,bbox(1)+bbox(3)+50];
            if(box_range(1)<=0) box_range(1)=0;end
            if(box_range(2)>size(videoFrame,1)) box_range(2)=size(videoFrame,1);end
            if(box_range(3)<=0) box_range(3)=0;end
            if(box_range(4)>size(videoFrame,2)) box_range(4)=size(videoFrame,2);end
% Create a MATLAB movie struct from the video frames.
for k = 1 : numFrames
    mov(k).cdata = imresize(vidFrames(box_range(1):box_range(2),box_range(3):box_range(4),:,k),3);
    mov(k).colormap = [];
end
hf = figure; 
% Resize figure based on the video's width and height
set(hf, 'position', [2250 -100 size(mov(1).cdata,1) size(mov(1).cdata,2)])
movie(hf, mov, [1 1 2 3], readerobj.FrameRate); 
fprintf('press any key to continue..')
pause
close(hf)
delete(readerobj)
clear('mov')
clear('readerobj')
release(faceDetector)
%10sec to allow ppl to leave the room
pause(10)

tic
for s_ind=9:34
    
    %if recording videos, skip speaker Nr.21 coz theres no video for this
    %speaker
    if(s_ind==21) 
      continue
    end

    speaker_name=['s', num2str(s_ind)];
    %first 3 are not video files
    video_list=ls([rootpath speaker_name]);
    
    video_list=video_list(4:end,:);
    for v_ind=1:size(video_list,1)
        if(rand>dropout)

            filename=[dest_dir speaker_name '-' video_list(v_ind,1:end-4)];
            % Construct a multimedia reader object associated with filename 
           readerobj = VideoReader([rootpath speaker_name '\' video_list(v_ind,:)]);

           % Read in all video frames.
           vidFrames = read(readerobj);

           % Get the number of frames.
           %numFrames = get(readerobj, 'NumberOfFrames');
            numFrames = size(vidFrames,4);
            videoFrame      = vidFrames(:,:,:,1);
            bbox = step(faceDetector, videoFrame);
            if(isempty(bbox)) continue; end
            box_range=[bbox(2)-40,bbox(2)+bbox(4)+40,bbox(1)-50,bbox(1)+bbox(3)+50];
            if(box_range(1)<=1) box_range(1)=1;end
            if(box_range(2)>size(videoFrame,1)) box_range(2)=size(videoFrame,1);end
            if(box_range(3)<=1) box_range(3)=1;end
            if(box_range(4)>size(videoFrame,2)) box_range(4)=size(videoFrame,2);end
           % Create a MATLAB movie struct from the video frames.
           for k = 1 : numFrames
                 mov(k).cdata = imresize(vidFrames(box_range(1):box_range(2),box_range(3):box_range(4),:,k),3);
                 mov(k).colormap = [];
           end
           
            % Create a cascade detector object.
            faceDetector = vision.CascadeObjectDetector();

            % Read a video frame and run the detector.
%             videoFileReader = vision.VideoFileReader('visionface.avi');
%             videoFrame      = step(videoFileReader);
%             bbox            = step(faceDetector, videoFrame);

           % Create a figure
           hf = figure; 

           % Resize figure based on the video's width and height
           set(hf, 'position', [2250 -100 size(mov(1).cdata,1) size(mov(1).cdata,2)])
           movie(hf, mov, [1 1 2 3], readerobj.FrameRate); 
           %start recording
            %if only 1 sensor:
            %commandJAER(u2,'zerotimestamps')
            pause(0.5)
           %commandJAER(u2,['startlogging ' filename])
            
            %commandJAER(handles.u1,['startlogging ' filename '\cochlea'])
    %         commandJAER(handles.u1,'zerotimestamps')
    %         commandJAER(handles.u1,'togglesynclogging')
           
           % Playback movie once at the video's frame rate

           movie(hf, mov, 1, readerobj.FrameRate);           
            %commandJAER(u2,['stoplogging ' filename '\DVS'])
            pause(0.2)
               %         commandJAER(handles.u1,'togglesynclogging')
    %         movefile([default_jpath 'Cochlea*'], [filename]);
    %         movefile([default_jpath 'DAVIS240C*'], [filename]);
    %         movefile([default_jpath 'JAERViewer*'], filename)
           close(hf)
           delete(readerobj)
           clear('readerobj')
           clear('mov')
            release(faceDetector)


        end
    end
end
toc
%fclose(u1);
fclose(u2);
