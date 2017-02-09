rng('shuffle');
align_path='C:\Users\YY\polybox\thesis\grid\align';
%file_list=ls(align_path);
%file_list=file_list(3:end-1,:);
load('file_list.mat');

number_of_sentences=10;
sen_ind=randperm(size(file_list,1),number_of_sentences);

sentences=file_list(sen_ind,1:6);
for i=1:length(sen_ind)
    sentence_full=short_sentence_to_full(sentences(i,:));
    %label=short_sentence_to_bow(sentences(i,:));
    %write to text file
    fileID = fopen('celldata.txt','a');
    formatSpec = '%s_%s_%s_%s_%s_%s\n';
    [nrows,ncols] = size(sentence_full);
    for row = 1:nrows
        fprintf(fileID,formatSpec,sentence_full{row,:});
    end
    fclose(fileID);
    type celldata.txt
end



