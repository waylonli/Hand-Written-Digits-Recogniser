function task2_8(Xtrain, Ytrain, Xtest, Ytest, epsilon, L)
% Input:
%   Xtrain : M-by-D training data matrix (double)
%   Xtrain : M-by-1 label vector for Xtrain (uint8)
%   Xtest  : N-by-D test data matrix (double)
%   Ytest  : N-by-1 label vector for Xtest (uint8)
%   epsilon : A scalar parameter for regularisation
%   L      : scalar (integer) of the number of Gaussian distributions per class
    display(sprintf('\n=== L = %g ===', L));
    % Get the running time of the new gaussian classifier
    tic
    [Ypreds MMs MCovs] = run_mgcs(Xtrain, Ytrain, Xtest, epsilon,L);
    toc
    D = length(Xtrain(1,:));
    % Get the first class mean vector and covariance matrix according to the task requirement
    Ms1 = MMs(1:L,:);
    Covs1 = MCovs(1:L,:,:);
%     save(['task2_8_g' num2str(L) '_m1.mat'],'Ms1');
%     save(['task2_8_g' num2str(L) '_cov1.mat'],'Covs1');
 
    % Compute the confusion matrix and display the information
    [CM, acc] = comp_confmat(Ytest, Ypreds);
%     save(['task2_8_cm_' num2str(L) '.mat'], 'CM');
    error = (1 - acc) * length(Ytest);
    display(sprintf('N = %g \nNerrs = %g \nacc = %g\n',length(Ytest), error, acc));
    
end
