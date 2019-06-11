function [Ypreds, MMs, MCovs] = run_mgcs(Xtrain, Ytrain, Xtest, epsilon, L)
% Input:
%   Xtrain : M-by-D training data matrix (double)
%   Ytrain : M-by-1 label vector for Xtrain (uint8)
%   Xtest  : N-by-D test data matrix (double)
%   epsilon : A scalar parameter for regularisation (double)
%   L      : scalar (integer) of the number of Gaussian distributions per class
% Output:
%  Ypreds : N-by-1 matrix of predicted labels for Xtest (integer)
%  MMMs     : (L*K)-by-D matrix of mean vectors (double)
%  MCovs   : (L*K)-by-D-by-D 3D array of covariance matrices (double)
    [M D] = size(Xtrain);
    N = length(Xtest(:,1));
    Ytrain_new = zeros(M,1);
    MMs = zeros(10*L,D);
    MCovs = zeros(10*L,D,D);
    % Pc is P(c), the probability of the class
    Pc = zeros(1,10);
    
    for i = 1:10
        % Pick out the elements with the same label
        Xtrn_new = Xtrain(Ytrain == i-1,:);
        % Apply kMeans to get L centres for each class
        [Centres idx ~] = my_kMeansClustering(Xtrn_new, L, Xtrn_new(1:L,:));
        temp = Ytrain == (i-1);
        Ytrain_new(temp,1) = L * (i-1) + idx;
    end
    
    for i = 1:10*L
        % Pick out the elements with the same label
        Xtrn_new = Xtrain(Ytrain_new == i,:);
        % Calculate the mean vector
        MMs(i,:) = my_mean(Xtrn_new);
        Cov = zeros(D,D);
        for j = 1:size(Xtrn_new(:,1))
            Cov = Cov + ((Xtrn_new(j,:) - MMs(i,:))')*(Xtrn_new(j,:) - MMs(i,:))/length(Xtrn_new(:,1));
        end
        % This line has to be added because logdet() function only accept
        % positive definite matrix as the argument
        MCovs(i,:,:) = Cov + eye(D)*epsilon;
        Pc(1,i) = log(length(Xtrn_new(:,1))/length(Xtrain(:,1)));
    end
    
    like = zeros(10*L,N);
    for i = 1:10*L
       mu = MMs(i,:); 
       sigma = reshape(MCovs(i,:,:),D,D);
       x = Xtest - repmat(mu, N, 1);
       % Calculate the likelihood using the formula
       temp = -1/2 .* x * inv(sigma) * x' -1/2 .* logdet(sigma) + Pc(1,i);
       like(i,:) = diag(temp);
    end
    % Choose the greatest probability
    [~, Ypreds] = max(like', [], 2);
    % Convert the classification to 0-9 labels
    Ypreds = ceil(Ypreds/L);
    Ypreds = Ypreds - 1;
end

function my_mean = my_mean(X)
    [row col] = size(X);
    sum = zeros(1,col);
    for i = 1:row
       sum = sum(1,:) + X(i,:); 
    end
    my_mean = sum./row; 
end