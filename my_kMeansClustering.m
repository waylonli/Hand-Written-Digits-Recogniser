%
function [C, idx, SSE] = my_kMeansClustering(X, k, initialCentres, maxIter)
% Input
%   X : N-by-D matrix (double) of input sample data
%   k : scalar (integer) - the number of clusters
%   initialCentres : k-by-D matrix (double) of initial cluster centres
%   maxIter  : scalar (integer) - the maximum number of iterations
% Output
%   C   : k-by-D matrix (double) of cluster centres
%   idx : N-by-1 vector (integer) of cluster index table
%   SSE : (L+1)-by-1 vector (double) of sum-squared-errors

  %% If 'maxIter' argument is not given, we set by default to 500
  if nargin < 4
    maxIter = 500;
  end
  
  %% TO-DO
  [row col] = size(X);
  C_pre = initialCentres;
  C = zeros(k,col);
  idx = nan;
  D = zeros(k,row);
  for i = 1:maxIter
     if i == 1
         SSE = zeros(1,1);
     else
         SSE = [SSE;0];
     end
     % Get all the Euclidean distance between X and the previous centres
     % Get the minimum index of distance as well
     [D,Ds,idx] = compute_dis(X,C_pre,k);
     
     for c = 1:k
         if sum(idx==c) == 0
            warn('k-means: cluster %d is empry',c);
         else
            % Get the new centres by calculate the mean vectors
            C(c,:) = my_mean(X(idx==c,:));
            SSE(i,1) = sum(Ds(1,:));
         end
     end
     error = 0;
     % Calculate the difference between the new centres and the previous centres
     % If the new centres and the last centres are the same, we are done!
     for c = 1:k
         error = error + sum(abs(C(c,:) - C_pre(c,:)));
     end
     if error == 0
         [D,Ds,idx] = compute_dis(X,C,k);
         SSE = [SSE;0];
         SSE(i+1,1) = sum(Ds(1,:));
         break;
     end
     C_pre = C;
  end
end

function sq_dist = square_dist(U, v)
    sq_dist = sum(bsxfun(@minus, U, v).^2, 2)';
end

function my_mean = my_mean(X)
    [row col] = size(X);
    sum = zeros(1,col);
    for i = 1:row
       sum = sum(1,:) + X(i,:); 
    end
    my_mean = sum./row; 
end

function [D,Ds,idx] = compute_dis(X,C,k)
    for c = 1:k
        D(c,:) = square_dist(X, C(c,:)); 
    end
    [Ds, idx] = min(D);
end