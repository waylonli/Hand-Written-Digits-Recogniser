%
%
function task2_5(Xtrain, Ytrain, Xtest, Ytest, epsilon)
% Input:
%  Xtrain : M-by-D training data matrix (double)
%  Ytrain : M-by-1 label vector for Xtrain (unit8)
%  Xtest  : N-by-D test data matrix (double)
%  Ytest  : N-by-1 label vector for Xtest (unit8)
%  epsilon : a scalar variable (double) for covariance regularisation
    
    % Get the running time of gaussian classifier
    tic
    [Ypreds Ms Covs] = run_gaussian_classifiers(Xtrain, Ytrain, Xtest, epsilon);
    toc
    D = length(Xtrain(1,:));
    % Get the 10th mean vector and covariance matrix according to the task requirement
    m_ten = Ms(10,:);
    cov_ten = reshape(Covs(10,:,:),D,D);
%     save('task2_5_m10.mat','m_ten');
%     save('task2_5_cov10.mat','cov_ten')
    
    % Compute the confusion matrix and display the information
    [CM, acc] = comp_confmat(Ytest, Ypreds);
%    save('task2_5_cm.mat','CM')
    error = (1 - acc) * length(Ytest);
    display(sprintf('N = %g \nNerrs = %g \nacc = %g\n',length(Ytest), error, acc));
    
end
