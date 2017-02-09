function [mfccs_third] = get_mfccs_for_file(filename)

[ts,fs] = audioread(filename);

% Define variables
Tw = 45;                % analysis frame duration (ms)
Ts = 30;                % analysis frame shift (ms)
alpha = 0.97;           % preemphasis coefficient
M = 20;                 % number of filterbank channels 
C = 12;                 % number of cepstral coefficients
L = 22;                 % cepstral sine lifter parameter
LF = 300;               % lower frequency limit (Hz)
HF = 3700;              % upper frequency limit (Hz)

% Read speech samples, sampling rate and precision from file
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
mfccs_third = single([MFCCs(:, 3:end); d1(2:end, :)'; d2'])';

end