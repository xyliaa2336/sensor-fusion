%% Build worklist
addpath(genpath('~/Dropbox/tools/various_speech/'));
AUDIO_DIR = '/home/dneil/datasets/grid/audio';
TRANSCRIPTION_DIR = '/home/dneil/datasets/grid/transcriptions/align';
wav_list = {};
identifier = {};
save_key = {};

idx = 1;
person = dir(AUDIO_DIR);
for person=person'
    if(person.isdir && ~strcmp(person.name,'.') && ~strcmp(person.name,'..'))
        audio_files = dir([AUDIO_DIR '/' person.name '/*.wav']);    
        for audio_file=audio_files'
            filename = audio_file.name(1:end-4);
            wav_list{idx} = [AUDIO_DIR '/' person.name '/' filename '.wav'];
            identifier{idx} = filename;
            save_key{idx} = [person.name '-' filename];
            idx = idx + 1;
        end
    end
end
fprintf('Done building worklist.\n');
%% Build MFCCs
output_feats = {};
% Define variables
Tw = 60;                % analysis frame duration (ms)
Ts = 40;                % analysis frame shift (ms)
alpha = 0.97;           % preemphasis coefficient
M = 20;                 % number of filterbank channels 
C = 12;                 % number of cepstral coefficients
L = 22;                 % cepstral sine lifter parameter
LF = 300;               % lower frequency limit (Hz)
HF = 3700;              % upper frequency limit (Hz)

fs = 50000;             % Sampling rate
for idx=1:numel(wav_list)
    % Read speech samples, sampling rate and precision from file
    [y, fs] = audioread(wav_list{idx});
    
    %ts = single(wav_list{idx});
    ts = single(y);
    speech=(ts-min(ts))*2/(max(ts)-min(ts))-1; 

    % Feature extraction (feature vectors as columns)
    [ MFCCs, FBEs, frames ] = ...
                    mfcc( speech, fs, Tw, Ts, alpha, @hamming, [LF HF], M, C+1, L );

    % Get times
    [ Nw, NF ] = size( frames );                % frame length and number of frames
    time_frames = [0:NF-1]*Ts*0.001+0.5*Nw/fs;  % time vector (s) for frames                 
    
    % Build MFCCs with derivs                
    % Get derivs
    d1 = diff(MFCCs');
    d2 = diff(diff(MFCCs'));
    
    output_times{idx} = time_frames;
    output_feats{idx} = single([MFCCs(:, 3:end); d1(2:end, :)'; d2']);
    output_label{idx} = save_key{idx};
end
fprintf('Done.');
% Save
save(['grid_mfccs_' num2str(Ts) '.mat'], 'output_feats', 'output_label', 'output_times');
fprintf('Saved.\n');
%% Demo
idx=1;
imagesc( output_times{idx}, 1:size(output_feats{idx},1), output_feats{idx});
axis('xy');
xlim([min(output_times{idx}) max(output_times{idx})]);
xlabel('Time (s)'); 
ylabel('Cepstrum index');
title(sprintf('Mel frequency cepstrum with first and second derivatives: %s', save_key{idx}));
colormap(1-colormap('gray'));
