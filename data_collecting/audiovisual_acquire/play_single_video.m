function play_single_video(filename)

       % Construct a multimedia reader object associated with filename 
       readerobj = VideoReader(filename);
 
       % Read in all video frames.
       vidFrames = read(readerobj);
 
       % Get the number of frames.
       numFrames = get(readerobj, 'NumberOfFrames');
 
       % Create a MATLAB movie struct from the video frames.
       for k = 1 : numFrames-1
             mov(k).cdata = vidFrames(:,:,:,k);
             mov(k).colormap = [];
       end
 
       % Create a figure
       hf = figure; 
       
       % Resize figure based on the video's width and height
       set(hf, 'position', [2000 300 readerobj.Width readerobj.Height])
       movie(hf, mov, [1 1 2], readerobj.FrameRate); 
       pause(0.2)
       % Playback movie once at the video's frame rate
       
       movie(hf, mov, 1, readerobj.FrameRate);
       pause(0.2)
       close(hf)

end