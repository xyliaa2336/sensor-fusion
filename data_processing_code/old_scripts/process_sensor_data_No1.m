testpath='F:\lipreading_data\sentences_cochlp_all\005\';
%trialname=1:9;
for trial_cnt=1:3
    %trialname_full=trialname(trial_cnt,:);
    trialname_full=[num2str(trial_cnt)];
    sentence_folder=ls([testpath, trialname_full]);
    file_list=textread([testpath, trialname_full '\celldata.txt'],'%s','delimiter', '\n');
    
    for i=1:length(file_list)
        pathname=[testpath,trialname_full,'\',strtrim(file_list{i}),'\']
        
        remain=strtrim(file_list{i});
        word=[];
        while true
           [str, remain] = strtok(remain, '_');
           if isempty(str),  break;  end
           word=[word str(1)];
        end
        des_dir=[testpath,'mats_beep'];
        mkdir(des_dir);
        get_word_from_recording_beep(pathname,word,des_dir);
        i
        
        close all;
    end
end
