rootpath='F:\hw_grid_mat\CochLP_ctime_40_on\';
name_list=ls([rootpath,'*.mat']);
for i=1: length(name_list)
    load([rootpath,name_list(i,:)]);
    break
end

%coch_mean= mean(reshape(coch_vect,[1,size(coch_vect,1)*size(coch_vect,2)]));
%coch_std=std(reshape(coch_vect,[1,size(coch_vect,1)*size(coch_vect,2)]));
coch_mean=mean(coch_vect);
coch_std=std(coch_vect);

coch_vect_norm=(coch_vect-ones(64,1)*coch_mean)./(ones(64,1)*coch_std);
%coch_vect_norm=(coch_vect-coch_mean)/coch_std;