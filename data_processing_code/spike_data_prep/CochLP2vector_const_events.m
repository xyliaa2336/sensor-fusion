%make cochlea events binned vectors
rootpath='F:\hw_grid_mat\CochLP_events_100_on\';

coch_list=ls(['F:\hw_grid_mat\','*_cochLP.mat']);
num_events=100;
number_of_channels=64;

for i=1:size(coch_list,1)
   load(['F:\hw_grid_mat\',coch_list(i,:)]);
   time_steps=1:floor(length(tsleftoff)/num_events)+1;
   if(mod(i,1000)==0)
		i
   end
		
%    if(length(time_steps)>1000)
%        fprintf('abnormal file at %i',i);
%        movefile(['F:\hw_grid_mat\',ret_list(i,:)],['F:\hw_grid_mat\abnormal_files\',ret_list(i,:)]);
%        movefile(['F:\hw_grid_mat\',ret_list(i,1:9),'_cochLP.mat'],['F:\hw_grid_mat\abnormal_files\',ret_list(i,1:9),'_cochLP.mat']);
%        continue
%    end
   coch_vect=zeros(number_of_channels,length(time_steps));
   
   for j=1:length(time_steps)
       window_start=num_events*(j-1)+1;
       if(window_start>length(tsleftoff)) 
           break
       end

       ind=window_start:min(window_start+num_events-1,length(tsleftoff));
           for ind_cnt=1:length(ind)
                    coch_vect(addrloff(ind(ind_cnt))+1,time_steps(j))=coch_vect(addrloff(ind(ind_cnt))+1,time_steps(j))+1;
           end
       
   end
   save([rootpath,coch_list(i,1:9),'_cochvector.mat'],'coch_vect');

end

          
       