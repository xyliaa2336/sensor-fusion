%make DVS binned frames
rootpath='F:\DVS_JAER_test\DVS_frames_test\';

ret_list=ls(['F:\DVS_JAER_test\','*_ret.mat']);
time_window=20e-3;
xrange=[69,188];
yrange=[0,119];
dummy=0;
for i=1:size(ret_list,1)
   load(['F:\DVS_JAER_test\',ret_list(i,:)]);
   allTs_ret=double(allTs_ret)/1e6;
   time_steps=1:floor((max(allTs_ret)-min(allTs_ret))/time_window);
   if(length(time_steps)>1000)
       fprintf('abnormal file at %i',i);
       movefile(['F:\DVS_JAER_test\',ret_list(i,:)],['F:\hw_grid_mat\abnormal_files\',ret_list(i,:)]);
       movefile(['F:\DVS_JAER_test\',ret_list(i,1:9),'_cochLP.mat'],['F:\hw_grid_mat\abnormal_files\',ret_list(i,1:9),'_cochLP.mat']);
       continue
   end
   %window_start=min(allTs_ret);
   DVS_frames=zeros(yrange(2)-yrange(1)+1,xrange(2)-xrange(1)+1,length(time_steps));
   DVS_frames_resized=zeros(48,48,length(time_steps));
   for j=1:length(time_steps)
       window_start=(min(allTs_ret)+time_window*(j-1));
       ind=find(allTs_ret>=window_start & allTs_ret<window_start+time_window);
       
       for ind_cnt=1:length(ind)
           if(x(ind(ind_cnt))>=xrange(1) && x(ind(ind_cnt))<=xrange(2) && y(ind(ind_cnt))>=yrange(1) && y(ind(ind_cnt))<=yrange(2))
               dummy=dummy+1;
                DVS_frames(y(ind(ind_cnt))+1-yrange(1),x(ind(ind_cnt))+1-xrange(1),time_steps(j))=DVS_frames(y(ind(ind_cnt))+1-yrange(1),x(ind(ind_cnt))+1-xrange(1),time_steps(j))+1;
           end
           
       end
       DVS_frames(:,:,time_steps(j))=flipud(DVS_frames(:,:,time_steps(j)));
       DVS_frames_resized(:,:,time_steps(j))=imresize(DVS_frames(:,:,time_steps(j)),[48,48],'Antialiasing',0,'Method','box');
   end
   save([rootpath,ret_list(i,1:9),'_DVSframes.mat'],'DVS_frames');
   save([rootpath,ret_list(i,1:9),'_DVSframes_resized.mat'],'DVS_frames_resized');
   clear DVS_frames;
   clear DVS_frames_resized;
   
end

          
       