video_names=ls('*_video.mat');
for file_cnt=1:size(video_names,1)
    close all;
    load(video_names(file_cnt,:));
    
    video_len=size(frame_final,3);
    frame_resized=zeros(48,48,video_len);
    for i=1:video_len
        frame_resized(:,:,i)=imresize(frame_final(:,30:end-31,i),[48,48]);
        %imagesc(frame_resized(:,:,i));axis('equal')
        %(0.1)
    end
    save([video_names(file_cnt,1:end-4),'_resized.mat'],'frame_resized','allTsd_frames')
end