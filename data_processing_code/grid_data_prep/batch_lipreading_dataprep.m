function [mfccs_batch, vid_batch, bY_batch] = batch_lipreading_dataprep(file_keys, data_dirs, blankout)

%audio_folder = '/home/dneil/datasets/grid/audio/';
%video_folder = '/home/dneil/datasets/grid/video/mat_data/';
audio_folder = data_dirs{1};
video_folder = data_dirs{2};

likelihood_of_audio_dropout = 0.8;

mfccs = cell(1,numel(file_keys));
vids = cell(1,numel(file_keys));
bY = cell(1, numel(file_keys));
max_len = 0;
for idx=1:numel(file_keys)
    % Get filenames
    split_string = strsplit(file_keys{idx}, '-');
    audio_filename = [audio_folder split_string{1} '/' split_string{2} '.wav'];
    video_filename = [video_folder file_keys{idx} '_vid.mat'];    
    
    % We only blankout after the first one, so that we know how big they
    % should be.  Could also pass it in as an argument.
    if(idx > 1 && rand()<blankout)
        if rand() < likelihood_of_audio_dropout
            mfccs{idx} = zeros(max_len, size(mfccs{idx-1}, 2));
            vids{idx} = get_vid_for_file(video_filename);
        else            
            mfccs{idx} = get_mfccs_for_file(audio_filename);
            vids{idx} = zeros(max_len, ...
                size(vids{idx-1},2), size(vids{idx-1},3), size(vids{idx-1},4));
        end
    else
        mfccs{idx} = get_mfccs_for_file(audio_filename);
        vids{idx} = get_vid_for_file(video_filename);        
    end
    
    % Get target
    bY{idx} = short_sentence_to_bow(split_string{2});
    max_len = max([max_len, size(mfccs{idx},1), size(vids{idx},1)]);
end

% Pad
mfccs_batch = single(zeros(numel(file_keys), max_len, size(mfccs{1},2)));
vid_batch = single(zeros(numel(file_keys), max_len, ...
    size(vids{1},2), size(vids{1},3), size(vids{1},4)));
bY_batch = single(zeros(numel(file_keys), size(bY{1}, 2)));
for idx=1:numel(mfccs)
    mfccs_batch(idx,end-size(mfccs{idx},1)+1:end,:) = mfccs{idx};
    vid_batch(idx,end-size(vids{idx},1)+1:end,:,:,:) = vids{idx};
    bY_batch(idx, :) = bY{idx};
end
end