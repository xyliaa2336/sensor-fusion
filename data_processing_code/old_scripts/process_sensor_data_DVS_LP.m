DVS_path='C:\Users\YY\polybox\thesis\recording_files\hw_grid_DVS';
LP_path='';
addpath('C:\Users\YY\polybox\thesis\lipreading\strings\strings')
namelist=ls(DVS_path);
namelist=namelist(3:end,:);

for cnt=1:size(namelist,1)
    c=strsplit(namelist(cnt,:),'-');
    speaker_name=c{1};
    label=c{2}(1:6);
    get_word_from_recording_DVS_LP(speaker_name, label,DVS_path, LP_path);
    close all;
   
end


