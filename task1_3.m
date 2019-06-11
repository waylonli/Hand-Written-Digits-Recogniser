%
%
function [EVecs, EVals, CumVar, MinDims] = task1_3(X)
% Input:
%  X : M-by-D data matrix (double)
% Output:
%  EVecs, Evals: same as in comp_pca.m
%  CumVar  : D-by-1 vector (double) of cumulative variance
%  MinDims : 4-by-1 vector (integer) of the minimum number of PCA dimensions
%            to cover 70%, 80%, 90%, and 95% of the total variance.
    % Get the eigenvalue and eigenvectors
    [EVecs, EVals] = comp_pca(X);
    k = 1;
    for i = 1:784
        % Calculte the cumulative variance using eigenvalues
        CumVar(k,1) = sum(abs(EVals(1:k)));
        k = k + 1;
    end;
    percent = [0.7; 0.8; 0.9; 0.95];
    k = 1;
    for i = 1:784
        % Detect when the cumulative variance covers ratio of the total variance
        if (CumVar(i,1)/CumVar(784,1)) >= percent(k,1)
            MinDims(k,1) = i;
            k = k + 1;
            if k > 4
                break;
            end
        end
    end
    plot(CumVar);
    title('Task1.3: Cumulative variances');
    xlabel('Dimension');
    ylabel('Cumulative variances');
%     save('task1_3_evecs','EVecs');
%     save('task1_3_evals','EVals');
end
