function [Ypreds] = run_knn_classifier(Xtrain, Ytrain, Xtest, Ks)
% Input:
%   Xtrain : M-by-D training data matrix (double)
%   Ytrain : M-by-1 label vector for Xtrain (uint8)
%   Xtest : N-by-D test data matrix (double)
%   Ks   : 1-by-L vector of the numbers of nearest neighbours in Xtrain (integer)
% Output:
%  Ypreds : N-by-L matrix of predicted labels for Xtest (integer)
    [row_trn,col_trn] = size(Xtrain);
    [row_tst,col_tst] = size(Xtest);
    Ypreds = zeros(row_tst,length(Ks));
    distance = zeros(row_tst,row_trn);
    for i = 1:row_tst
        distance(i,:) = square_dist(Xtest(i,:),Xtrain); 
    end
    
    % Sort the distance in order to get the cloest k vectors
    [dis_sort, idx] = sort(distance, 2, 'ascend');

    for i = 1:length(Ks)
         j = Ks(1,i);
         knn = dis_sort(:,1:j); 
         % Choose the cloest k ones
         idx_k = idx(:,1:j);
         for k = 1:row_tst
            class = Ytrain(idx_k(k,:),1)';
            % Get the most frequent label of the cloest k vectors
            Ypreds(k,i) = mode(class) + 1;
         end
    end
    Ypreds = uint8(Ypreds);
    Ypreds = Ypreds - 1;
end

function sq_dist = square_dist(U, v)
    sq_dist = sum(bsxfun(@minus, U, v).^2, 2)';
end