%
%
function [Corrs] = task2_4(Xtrain, Ytrain)
% Input:
%  Xtrain : M-by-D data matrix (double)
%  Ytrain : M-by-1 label vector (unit8) for X
% Output:
%  Corrs  : (K+1)-by-1 vector (double) of correlation $r_{12}$ 
%           for each class k = 1,...,K, and the last element holds the
%           correlation for the whole data, i.e. Xtrain.
    
    [EVecs, EVals] = comp_pca(Xtrain);
    principals = EVecs(:,1:2);
    X_2D = Xtrain * principals;
    Cov = zeros(2,2);
    Corrs = zeros(11,1);
    for num = 1:10
        % Pick out the vectors with the same label
        X_num = X_2D(Ytrain == (num-1),:);
        x_mean = zeros(1,2);
        for i = 1:2
            a = sum(X_num(:,i));
            b = length(X_num(:,1));
            x_mean(1,i) = a / b;
        end
        % Calculate the covariance matrix for each class
        for i = 1:size(X_num(:,1))
            Cov = Cov + ((X_num(i,:) - x_mean)')*(X_num(i,:) - x_mean)/b;
        end
        % Calculate the correlations for each class
        Corrs(num,1) = Cov(1,2) / sqrt(Cov(1,1)*Cov(2,2));
    end
    % Calculate the correlations for the whole data
    for i = 1:2
        a = sum(X_2D(:,i));
        b = length(X_2D(:,1));
        x_mean(1,i) = a / b;
    end
    for i = 1:size(X_2D(:,1))
            Cov = Cov + ((X_2D(i,:) - x_mean)')*(X_2D(i,:) - x_mean)/b;
    end
    Corrs(11,1) = Cov(1,2) / sqrt(Cov(1,1)*Cov(2,2));
%     save('task2_4_corrs.mat','Corrs');
end
