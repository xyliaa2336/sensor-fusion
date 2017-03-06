%%
clear all;close all; clc
pathname='D:\lipreading_data\sentences_cochlp_all\005\';

fprintf('...loading data...\n')
for i=1:3
    d = dir([pathname,num2str(i)]);
    isub = [d(:).isdir]; %# returns logical vector
    nameFolds = {d(isub).name}';
    nameFolds(ismember(nameFolds,{'.','..'})) = [];
    for j=1:length(nameFolds)
        DVS_name=ls([pathname,num2str(i),'\',nameFolds{j},'\DAVIS*']);
        [x,y,pol,allTs_ret] = getDVSeventsDavis([pathname,num2str(i),'\',nameFolds{j},'\',DVS_name]); 
        shortnames= strsplit(nameFolds{j},'_');
        shortname='';
        for k=1:length(shortnames)
            shortname=[shortname,shortnames{k}(1)];
        end
        save([pathname,'\mats\',shortname,'_ret.mat'],'x','y','pol','allTs_ret');
    end
end


