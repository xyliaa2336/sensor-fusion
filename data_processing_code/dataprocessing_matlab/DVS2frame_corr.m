%make DVS binned frames, from spikes correlated with cochlea spikes
dvs_path='F:\lipread_data\hw_grid_mat\';
coch_path='F:\HWSpikesCochLP_Grid\cochLP_mat\';
rootpath='F:\lipread_data\correlated_spikegrams\';
ret_list=ls([dvs_path,'s*.mat']);
size(ret_list,1)

%% get correlated spikes
time_window=40e-3;

xrange=[39,218];
yrange=[0,179];
dummy=0;
for i=5096:size(ret_list,1)
    if(mod(i,100)==0)
        i
    end
   load([dvs_path,ret_list(i,:)]);
   allTs_ret=double(allTs_ret)/1e6-0.5;
   ind=find(allTs_ret>0);
   allTs_ret=allTs_ret(ind);
   x=x(ind);y=y(ind);pol=pol(ind);
   time_steps=1:floor((max(allTs_ret)-min(allTs_ret))/time_window);
   
   name=[ret_list(i,1:end-8),'.mat'];
   if(length(time_steps)>1000)
       fprintf('abnormal file at %i',i);
       movefile([dvs_path,ret_list(i,:)],[dvs_path,'abnormal_files\',ret_list(i,:)]);
       %movefile(['D:\lipread_data\hw_grid_mat\',ret_list(i,1:9),'_cochLP.mat'],['F:\hw_grid_mat\abnormal_files\',ret_list(i,1:9),'_cochLP.mat']);
       continue
   end
   try
       load([coch_path,name]);
   catch 
       fprintf('cochlea file doesnt exist');
       continue
   end
   tsleft=tslefton;
   laddress=addrlon;
   
   cochlea_ts = tsleft;
    all_bins = 0:time_window:max(cochlea_ts);

   
   %%remove 1st Ts of each pixel, removes noise
    maxy=max(y); maxx=max(x); mat_times=zeros(size(allTs_ret,1)+1, maxy+1)+7;
    for k=1:length(y)
        indx=x(k)+1; indy=y(k)+1; timeTemp=allTs_ret(k);
        if timeTemp < mat_times(k)
            mat_times(indx,indy) = timeTemp;
        end
    end

    indm=find(mat_times<7); mat_times_reduced=mat_times(indm);
    indr=find(~ismember(allTs_ret,mat_times_reduced)==1);

    xr=x(indr); yr=y(indr); polr=pol(indr); allTsr=allTs_ret(indr);
 
 %%remove fast spiking pixels
    mxr=xr; myr=yr; mpolr=polr; mallTs=allTsr; mat2d=zeros(maxx, maxy);
    min2d=zeros(maxx, maxy); var2d=zeros(maxx, maxy); maxcount=0; 

    for k=1:maxy %length(myr)
        for j=1:maxx %length(mxr)
            indxr=find(mxr==j & myr==k);
            if (~isempty(indxr))
                timeTemp=allTsr(indxr); indcount=length(timeTemp); diffT=diff(timeTemp);
                mat2d(j,k)=indcount;
                min2d(j,k)=mean(diffT); 
                var2d(j,k)=var(double(diffT)); 
            end
    %         if indcount>maxcount
    %             max_d=min(diffT);
    %             max_y=k; max_x=j;
    %             maxcount=indcount;
    %         end
            %diffTemp=diff(timeTemp);indcount=sum(find(diffTemp<0.4e5));
            %if indcount>500
            %    k, j, diff(timeTemp),%mallTs(indxr)=9000000;
            %end
        end
    end
    %[row, col, val]=find(min2d>0.00001 & min2d<0.2e5 & mat2d>20);
    %[row, col, val]=find(mat2d>20);
    m2d=var2d./min2d;
    [row, col, val]=find(m2d <100 & mat2d>20);

    for l=1:length(col)
        indt=find(myr==col(l) & mxr==row(l)); mallTs(indt)=500;
    end

    indnew=find(mallTs~=500);
    
    %% Calculate spike rate of cochlea
    cochlea_ts = tsleft;
    all_bins = 0:time_window:max(cochlea_ts);
    cochlea_count = histc(cochlea_ts, all_bins);

    %% Calculate correlation of each pixel with the cochlea spike rate
    davis_x = mxr(indnew);
    davis_y = myr(indnew);
    davis_ts = mallTs(indnew);
    davis_pol=mpolr(indnew);
    flat_pixel_addr = davis_y*240+davis_x+1;
    pixels_to_check = unique(flat_pixel_addr);
    xcorr_vals = zeros(1, length(unique(flat_pixel_addr)));
    max_corr_series = [];
    best_pixel_timeseries = [];
    for ip=1:length(pixels_to_check)
        pixel_addr = pixels_to_check(ip);
        pixel_count_binned = histc(davis_ts(flat_pixel_addr==pixel_addr), all_bins);
        [pixel_xcorr, lag] = xcorr(pixel_count_binned, cochlea_count);
        xcorr_vals(ip) = pixel_xcorr(find(lag==0));
        if max(xcorr_vals)==xcorr_vals(ip)
           best_pixel_timeseries = pixel_count_binned;
           max_corr_series = abs(xcorr(pixel_count_binned, cochlea_count));
        end
    end

    threshold = 0.2;
    xcorr_vals_normd = xcorr_vals/max(xcorr_vals);
    pixel_corrs = zeros(1,240*180);
    pixel_corrs(pixels_to_check+1) = xcorr_vals_normd;
    above_thresh = pixel_corrs(davis_y*240+davis_x+1) > threshold;

    x=davis_x(above_thresh);y=davis_y(above_thresh);allTs_ret=davis_ts(above_thresh);
    pol=davis_pol(above_thresh);
    
    %%
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
   save([rootpath,ret_list(i,1:end-8),'_DVSframes.mat'],'DVS_frames');
   save([rootpath,ret_list(i,1:end-8),'_DVSframes_resized.mat'],'DVS_frames_resized');
   %normalization. Only on resized frames to save some time
   DVS_normed=DVS_frames_resized;
   flatten_DVS=reshape(DVS_frames_resized,[],1);
   flatten_DVS_NZ=flatten_DVS(flatten_DVS~=0);
   mean_nz=mean(flatten_DVS_NZ);std_nz=std(flatten_DVS_NZ);
   DVS_normed(DVS_frames_resized~=0)=DVS_frames_resized(DVS_frames_resized~=0)-mean_nz;
   DVS_normed(DVS_frames_resized~=0)=DVS_normed(DVS_frames_resized~=0)/std_nz;
   
   save([rootpath,ret_list(i,1:end-8),'_DVSframes_normed.mat'],'DVS_normed');
   
   clear DVS_frames;
   clear DVS_frames_resized;
   clear DVS_normed;
   
end
