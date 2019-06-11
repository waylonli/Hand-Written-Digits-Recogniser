%
%
function [CM, acc] = task2_7(Xtrain, Ytrain, Xtest, Ytest, epsilon, ratio)
% Input:
%  Xtrain : M-by-D training data matrix (double)
%  Ytrain : M-by-1 label vector for Xtrain (unit8)
%  Xtest  : N-by-D test data matrix (double)
%  Ytest  : N-by-1 label vector for Xtest (unit8)
%  ration : scalar (double) - ratio of training data to use.
% Output:
%  CM     : K-by-K matrix (integer) of confusion matrix
%  acc    : scalar (double) of correct classification rate
    [row col] = size(Xtrain);
    % Calculate the number of elements using ratio
    num = ratio * row;
    Xtrain = Xtrain(1:num,:);
    Ytrain = Ytrain(1:num,:);
    
    [Ypreds Ms Covs] = run_gaussian_classifiers(Xtrain, Ytrain, Xtest, 0.01);
    % Compute the confusion matrix
    [CM, acc] = comp_confmat(Ytest, Ypreds);
%     save(['task2_7_cm_' num2str(ratio*100)],'CM');
    
end
