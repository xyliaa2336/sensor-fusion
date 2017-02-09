function [vid] = get_vid_for_file(filename)
    load_data = load(filename);
    % Swap around for time x input_channels x row x col
    vid = permute(single(load_data.vid)/255, [1,4,2,3])-0.5;
    % Take the mean over the color dimension
    vid = mean(vid, 2);
end
