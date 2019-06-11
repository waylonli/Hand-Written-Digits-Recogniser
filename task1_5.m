%
%
function task1_5(X, Ks)
% Input:
%  X  : M-by-D data matrix (double)
%  Ks : 1-by-L vector (integer) of the numbers of nearest neighbours

    for i = 1:length(Ks)
        initialCentres = X(1:Ks(1,i),:);
        display(sprintf('\nk = %g',Ks(1,i)));
        % Get the running time of kmeans clustering
        tic
        [C, idx, SSE] = my_kMeansClustering(X, Ks(1,i), initialCentres);
        toc
%         save(['task1_5_c_' num2str(Ks(1,i))],'C');
%         save(['task1_5_idx_' num2str(Ks(1,i))],'idx');
%         save(['task1_5_sse_' num2str(Ks(1,i))],'SSE');
        % Length of SSE represents the x axis, SSE value is the y axis
        plot(1:length(SSE),SSE(:,1));
        title(['SSE for k = ' num2str(Ks(1,i))]);
        xlabel('Iteration number');
        ylabel('SSE');
        hold on;
    end
    legend('1','2','3','4','5','7','10','15','20');
end
