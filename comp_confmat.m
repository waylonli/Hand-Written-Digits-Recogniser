function [CM, acc] = comp_confmat(Ytrues, Ypreds)
% Input:
%   Ytrues : N-by-1 ground truth label vector
%   Ypreds : N-by-1 predicted label vector
% Output:
%   CM : K-by-K confusion matrix, where CM(i,j) is the number of samples whose target is the ith class that was classified as j
%   acc : accuracy (i.e. correct classification rate)
    % There are 10 classes in total
    K = 10;
    CM = zeros(K,K);
    row = length(Ytrues);
    % CM(i,j) represents element should be classfied to i but is classified to j
    for i = 1:row
        CM(Ytrues(i,1)+1,Ypreds(i,1)+1) = CM(Ytrues(i,1)+1,Ypreds(i,1)+1) + 1;
    end
    % diag(CM) represent the elements which are correctly classified
    correct = sum(diag(CM));
    acc = correct / row;
end
