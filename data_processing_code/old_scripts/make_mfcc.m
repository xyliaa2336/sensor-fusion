%% Build worklist
addpath('C:\Users\YY\polybox\thesis\mfcc')
audio_names=ls('*_audio.mat');

%% compute MFCC for all
% Define variables
Tw = 60;                % analysis frame duration (ms)
Ts = 40;                % analysis frame shift (ms)
alpha = 0.97;           % preemphasis coefficient
M = 20;                 % number of filterbank channels 
C = 12;                 % number of cepstral coefficients
L = 22;                 % cepstral sine lifter parameter
LF = 300;               % lower frequency limit (Hz)
HF = 3700;              % upper frequency limit (Hz)

fs = 48000;             % Sampling rate
output_feats = {};

for idx = 1:size(audio_names,1)
    if(mod(idx,100)==1)
        idx
    end
    load(audio_names(idx,:));
    ts = single(aud);
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
    output_label{idx} = audio_names(idx,1:6);
end
fprintf('Done.\n');
 %% save mfcc
 save(['recording_mfccs_' num2str(Ts) '.mat'], 'output_feats', 'output_label', 'output_times');
 %% Demo
idx=1;
imagesc( output_times{idx}, 1:size(output_feats{idx},1), output_feats{idx});
axis('xy');
xlim([min(output_times{idx}) max(output_times{idx})]);
xlabel('Time (s)'); 
ylabel('Cepstrum index');
title(sprintf('Mel frequency cepstrum with first and second derivatives: %s', output_label{idx}));
colormap(1-colormap('gray'));