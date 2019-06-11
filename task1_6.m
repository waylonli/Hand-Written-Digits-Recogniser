%
%
function task1_6(MAT_ClusterCentres)
% Input:
%  MAT_ClusterCentres : file name of the MAT file that contains cluster centres C.
%       
% 
    C = importdata(MAT_ClusterCentres);
    [row col] = size(C);
    for s = 1:row
        img(:,:,:,s) = reshape(C(s,:), 28, 28,1)';
    end
    montage(img);
    pause;
end
