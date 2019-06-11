function [EVecs, EVals] = comp_pca(X)
% Input: 
%   X:  N x D matrix (double)
% Output: 
%   EVecs: D-by-D matrix (double) contains all eigenvectors as columns
%       NB: follow the Task 1.3 specifications on eigenvectors.
%   EVals:
%       Eigenvalues in descending order, D x 1 vector (double)
%   (Note that the i-th columns of Evecs should corresponds to the i-th element in EVals)
  %% TO-DO
    
    EVecs = nan;
    EVals = nan;
    [row col] = size(X);
    S = zeros(col,col);
    % Calculate the mean vector of X
    for i = 1:col
       x_mean(1,i) = sum(X(:,i)) / row;
    end
    % Apply the formula to calculate covariance matrix
    for i = 1:row
        S = S + ((X(i,:) - x_mean)')*(X(i,:) - x_mean)/row;
    end
    % Get the eigenvectors and eigenvalue of covariance matrix
    [EVecs_temp,EVals_temp] = eig(S);
    % Let the first number of eigenvector is positive according to handout
    for i = 1:col
        if EVecs_temp(1,i) < 0
            EVecs_temp(:,i) = -EVecs_temp(:,i);
        end
    end
    EVecs = zeros(col,col);
    EVals = zeros(col,1);
    % Sort the eigenvalue and swap the position of eigenvectors
    for i = 1:col
        max_index = 0;
        max = -1;
        j = 1;
        % Get the greatest eigenvalue every time and put it at the front
        while j <= col
           if EVals_temp(j,j) > max
               max = EVals_temp(j,j);
               max_index = j;
           end
           j = j+1;
        end
        % Swap the eigenvectors as well
        EVals(i,1) = EVals_temp(max_index,max_index);
        EVecs(:,i) = EVecs_temp(:,max_index);
        EVals_temp(max_index,max_index) = -2;
    end
end

