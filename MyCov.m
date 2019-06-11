function cov = MyCov(X)
    [row col] = size(X);
    cov = zeros(col,col);
    % Calculate the mean vector of X
    x_mean = MyMean(X);
    % Apply the formula to calculate covariance matrix
    for i = 1:row
        cov = cov + ((X(i,:) - x_mean)')*(X(i,:) - x_mean)/row;
    end
end