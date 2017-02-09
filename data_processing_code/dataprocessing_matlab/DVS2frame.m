%make DVS binned frames
rootpath='F:\DVS_frames_40_norm_part4\';

ret_list=ls(['F:\hw_grid_DVS_part4_mat\','s*.mat']);
time_window=40e-3;
xrange=[39,218];
yrange=[0,179];
dummy=0;
for i=1:size(ret_list,1)
   load(['F:\hw_grid_DVS_part4_mat\',ret_list(i,:)]);
   allTs_ret=double(allTs_ret)/1e6-0.5;
   ind=find(allTs_ret>0);
   allTs_ret=allTs_ret(ind);
   x=x(ind);y=y(ind);pol=pol(ind);
   time_steps=1:floor((max(allTs_ret)-min(allTs_ret))/time_window);
   if(length(time_steps)>1000)
       fprintf('abnormal file at %i',i);
       movefile(['F:\hw_grid_DVS_part4_mat\',ret_list(i,:)],['F:\hw_grid_DVS_part4_mat\abnormal_files\',ret_list(i,:)]);
       %movefile(['D:\lipread_data\hw_grid_mat\',ret_list(i,1:9),'_cochLP.mat'],['F:\hw_grid_mat\abnormal_files\',ret_list(i,1:9),'_cochLP.mat']);
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
               if(pol(ind(ind_cnt))==0)
                   DVS_frames(y(ind(ind_cnt))+1-yrange(1),x(ind(ind_cnt))+1-xrange(1),time_steps(j))=DVS_frames(y(ind(ind_cnt))+1-yrange(1),x(ind(ind_cnt))+1-xrange(1),time_steps(j))-1;
               else
                   DVS_frames(y(ind(ind_cnt))+1-yrange(1),x(ind(ind_cnt))+1-xrange(1),time_steps(j))=DVS_frames(y(ind(ind_cnt))+1-yrange(1),x(ind(ind_cnt))+1-xrange(1),time_steps(j))+1;
               end
                   
                dummy=dummy+1;
                %DVS_frames(y(ind(ind_cnt))+1-yrange(1),x(ind(ind_cnt))+1-xrange(1),time_steps(j))=DVS_frames(y(ind(ind_cnt))+1-yrange(1),x(ind(ind_cnt))+1-xrange(1),time_steps(j))+1;
           end
           
       end
       DVS_frames(:,:,time_steps(j))=flipud(DVS_frames(:,:,time_steps(j)));
       DVS_frames_resized(:,:,time_steps(j))=imresize(DVS_frames(:,:,time_steps(j)),[48,48]);
   end
   save([rootpath,ret_list(i,1:10),'_DVSframes.mat'],'DVS_frames');
   save([rootpath,ret_list(i,1:10),'_DVSframes_resized.mat'],'DVS_frames_resized');
   %normalization. Only on resized frames to save some time
   DVS_normed=DVS_frames_resized;
   flatten_DVS=reshape(DVS_frames_resized,[],1);
   flatten_DVS_NZ=flatten_DVS(flatten_DVS~=0);
   mean_nz=mean(flatten_DVS_NZ);std_nz=std(flatten_DVS_NZ);
   DVS_normed(DVS_frames_resized~=0)=DVS_frames_resized(DVS_frames_resized~=0)-mean_nz;
   DVS_normed(DVS_frames_resized~=0)=DVS_normed(DVS_frames_resized~=0)/std_nz;
   
   save([rootpath,ret_list(i,1:10),'_DVSframes_normed.mat'],'DVS_normed');
   
   clear DVS_frames;
   clear DVS_frames_resized;
   clear DVS_normed;
   
end

          
       