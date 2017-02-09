%%
clear all;close all; clc
pathname='D:\HWSpikesCochLP_Grid\RecordingGridVol0d1\';

fprintf('...loading data...\n')

%%
folder_names=ls([pathname]);
folder_names=folder_names(3:end,:);

disp('Loading retina data file (events)...');
%[x,y,pol,allTs_ret] = getDVSeventsDavis([basefil '_ret.aedat']); % Spikes
for j =1:size(folder_names,1)
    if(j==11)
        continue;
    end
    filenames=ls([pathname strtrim(folder_names(j,:)) '\*.aedat']);
    filenames=filenames(3:end,:);
    speaker_ind=folder_names(j,15:15+length(strtrim(folder_names(j,:)))-33);
    for i=1:size(filenames,1)
        [allAddr,allTs]=loadaerdat([pathname,strtrim(folder_names(j,:)),'\',filenames(i,:)]); 
        [ tslefton, addrlon,tsleftoff,addrloff, tsrighton, addrron,tsrightoff,addrroff,allTsADC0,renorm_valueADC0, allTsADC1,renorm_valueADC1]=plotloadaeADCcochLP(allTs, allAddr);
        save(['D:\HWSpikesCochLP_Grid\cochLP_mat\',speaker_ind,'-',filenames(i,1:end-6),'.mat'],'tslefton','addrlon','tsleftoff','addrloff');
    end
    
end
