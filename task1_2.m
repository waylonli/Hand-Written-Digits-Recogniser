%
%
function M = task1_2(X, Y)
% Input:
%  X : M-by-D data matrix (double)
%  Y : M-by-1 label vector (unit8)
% Output:
%  M : (K+1)-by-D mean vector matrix (double)
%      Note that M(K+1,:) is the mean vector of X.
    MAT_M = zeros(11,784);
    [row col] = size(X);
    for k = 1:10
        j = 1;
        for i = 1:row
            % Select the elements with label (k-1)
            if (Y(i,1) == (k-1))
                MAT_M(k,:) = MAT_M(k,:) + X(i,:);
                j = j + 1;
            end
        end
        % Calculate the mean of each class
        MAT_M(k,:) = double(MAT_M(k,:)) / j;
    end
    % Calculate the mean vector for the whole Xtrain
    for i = 1:row
       MAT_M(11,:) = MAT_M(11,:) + X(i,:); 
    end
    MAT_M(11,:) = double(MAT_M(11,:)) / row;
    for s = 1:11
        M(:,:,:,s) = reshape(MAT_M(s,:), 28, 28,1)';
    end
%     save('task1_2_M','MAT_M');
    montage(M)
end
