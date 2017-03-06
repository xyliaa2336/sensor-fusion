%make cochlea events binned vectors
rootpath='D:\lipreading_data\sentences_cochlp_all\coch_40_on_normed\';

coch_list=ls(['D:\lipreading_data\sentences_cochlp_all\005\mats\','*_cochLP.mat']);
time_window=40e-3;
number_of_channels=64;


for i=1:size(coch_list,1)
	if(~(length(strtrim(coch_list(i,:))) == 17))
		continue;
	end
   load(['D:\lipreading_data\sentences_cochlp_all\005\mats\',strtrim(coch_list(i,:))]);
   time_steps=1:floor((max([tsleftoff;tslefton])-min([tsleftoff;tslefton]))/time_window);
   if(mod(i,1000)==0)
		i
   end
		
   if(length(time_steps)>1000)
       fprintf('abnormal file at %i',i);
       %movefile(['F:\hw_grid_mat\',ret_list(i,:)],['F:\hw_grid_mat\abnormal_files\',ret_list(i,:)]);
       %movefile(['D:\HWSpikesCochLP_Grid\cochLP_mat\',coch_list(i,1:9),'_cochLP.mat'],['F:\hw_grid_mat\abnormal_files\',ret_list(i,1:9),'_cochLP.mat']);
       continue
   end
   coch_vect=zeros(number_of_channels,length(time_steps));
   
   for j=1:length(time_steps)
       window_start=(min([tsleftoff;tslefton])+time_window*(j-1));
%        indoff=find(tsleftoff>=window_start & tsleftoff<window_start+time_window);
       indon=find(tslefton>=window_start & tslefton<window_start+time_window);
       
%        for ind_cnt=1:length(indoff)          
%           coch_vect(addrloff(indoff(ind_cnt))+1,time_steps(j))=coch_vect(addrloff(indoff(ind_cnt))+1,time_steps(j))-1;
%        end
%        
       for ind_cnt=1:length(indon)      
          coch_vect(addrlon(indon(ind_cnt))+1,time_steps(j))=coch_vect(addrlon(indon(ind_cnt))+1,time_steps(j))+1;           
       end

   end
    coch_mean= mean(reshape(coch_vect,[1,size(coch_vect,1)*size(coch_vect,2)]));
    coch_std=std(reshape(coch_vect,[1,size(coch_vect,1)*size(coch_vect,2)]));
    coch_vect_norm=(coch_vect-coch_mean)/coch_std;
    
    if(strcmp(coch_list(i,1:3),'s9-'))
		save([rootpath,coch_list(i,1:9),'_cochvector.mat'],'coch_vect');
	else
		save([rootpath,'rs5-', coch_list(i,1:6),'_cochvector.mat'],'coch_vect');
        save([rootpath,'rs5-', coch_list(i,1:6),'_cochvector_normed.mat'],'coch_vect_norm');
    end
		
   
end

          
       