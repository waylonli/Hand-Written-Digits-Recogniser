function [Ypreds, Ms, Covs] = run_gaussian_classifiers(Xtrain, Ytrain, Xtest, epsilon)
% Input:
%   Xtrain : M-by-D training data matrix (double)
%   Ytrain : M-by-1 label vector for Xtrain (uint8)
%   Xtest  : N-by-D test data matrix (double)
%   epsilon : A scalar variable (double) for covariance regularisation
% Output:
%  Ypreds : N-by-1 matrix (uint8) of predicted labels for Xtest
%  Ms     : K-by-D matrix (double) of mean vectors
%  Covs   : K-by-D-by-D 3D array (double) of covariance matrices

%YourCode - Bayes classification with multivariate Gaussian distributions.
    
    [row_xt col_xt] = size(Xtest);
    Ms = zeros(10,col_xt);
    Covs = zeros(10,col_xt,col_xt);
    logCovs = zeros(1,10);
    Ypreds = zeros(row_xt,1);
    % Pc is P(c), the probability of the class
    Pc = zeros(1,10);
    
    for i = 1:10
       % Pick out the elements with the same label
       class = Xtrain(Ytrain == (i-1),:);
       % Calculate the mean vector
       for j = 1:col_xt
          Ms(i,j) = sum(class(:,j)) / length(class(:,1));  
       end
       Cov = zeros(col_xt,col_xt);
       % Calculate the covariance matrix for each class
       for j = 1:length(class(:,1))
             Cov = Cov + ((class(j,:) - Ms(i,:))')*(class(j,:) - Ms(i,:))/length(class(:,1));
       end
       % This line has to be added because logdet() function only accept
       % positive definite matrix as the argument
       Covs(i,:,:) = Cov + eye(col_xt) * epsilon;
       Pc(1,i) = log(length(class(:,1))/length(Xtrain(:,1)));
    end
    
    
    like = zeros(10,row_xt);
    for i = 1:10
       mu = Ms(i,:); 
       sigma = reshape(Covs(i,:,:),col_xt,col_xt);
       x = Xtest - repmat(mu, row_xt, 1);
       % Calculate the likelihood using the formula
       temp = -1/2 .* x * inv(sigma) * x' -1/2 .* logdet(sigma) + Pc(1,i);
       like(i,:) = diag(temp);
    end
    
    % Choose the greatest probability
    [~, Ypreds] = max(like', [], 2);
    Ypreds = Ypreds - 1;
end
