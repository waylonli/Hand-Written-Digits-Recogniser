%
%
function task2_1(Xtrain, Ytrain, Xtest, Ytest, Ks)
% Input:
%  Xtrain : M-by-D training data matrix (double)
%  Ytrain : M-by-1 label vector (unit8) for Xtrain
%  Xtest  : N-by-D test data matrix (double)
%  Ytest  : N-by-1 label vector (unit8) for Xtest
%  Ks     : 1-by-L vector (integer) of the numbers of nearest neighbours in Xtrain

    % Get the running time of knn classifier
    tic
    [Ypreds] = run_knn_classifier(Xtrain, Ytrain, Xtest, Ks);
    toc

    for i = 1:length(Ks)
        [CM, acc] = comp_confmat(Ytest, Ypreds(:,i));
%         save(['task2_1_cm' num2str(Ks(1,i))],'CM');
        % error rate + accurate rate = 1
        error = (1 - acc) * length(Ytest);
        display(sprintf('\nk = %g \nN = %g \nNerrs = %g \nacc = %g',Ks(1,i),length(Ytest), error, acc));
    end
end
