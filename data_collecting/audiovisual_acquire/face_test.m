tic
readerobj = VideoReader('F:\grid\originals\video\s1\bbaf2n.mpg');

           % Read in all video frames.
           vidFrames = read(readerobj);

           % Get the number of frames.
           %numFrames = get(readerobj, 'NumberOfFrames');
            numFrames = size(vidFrames,4);
           
            % Create a cascade detector object.
            faceDetector = vision.CascadeObjectDetector();
toc
            % Read a video frame and run the detector.
            %videoFileReader = vision.VideoFileReader('visionface.avi');
            videoFrame      = vidFrames(:,:,:,1);
            bbox            = step(faceDetector, videoFrame);
            % Draw the returned bounding box around the detected face.
videoOut = insertObjectAnnotation(videoFrame,'rectangle',bbox,'Face');
figure, imshow(videoOut), title('Detected face');
            toc
            figure
            imagesc(videoFrame(bbox(2)-40:bbox(2)+bbox(4)+40,bbox(1)-40:bbox(1)+bbox(3)+40,:,:))
            
            axis equal
            
           % Create a MATLAB movie struct from the video frames.
           for k = 1 : numFrames
                 im=imresize(vidFrames(bbox(2)-40:bbox(2)+bbox(4)+40,bbox(1)-50:bbox(1)+bbox(3)+50,:,k),3);
                 mov(k).cdata = im;
                 mov(k).colormap = [];
           end
            hf = figure; 
            
           % Resize figure based on the video's width and height
            set(hf, 'position', [2250 -100 size(im,1) size(im,2)])
           movie(hf, mov, [1 4 5], readerobj.FrameRate); 
           
           %start recording
            %if only 1 sensor:
%             commandJAER(u2,'zerotimestamps')
            pause(0.5)
%            commandJAER(u2,['startlogging ' filename])
            
            %commandJAER(handles.u1,['startlogging ' filename '\cochlea'])
    %         commandJAER(handles.u1,'zerotimestamps')
    %         commandJAER(handles.u1,'togglesynclogging')
           
           % Playback movie once at the video's frame rate

           movie(hf, mov, 1, readerobj.FrameRate);     