%
%
function task1_4(EVecs)
% Input:
%  Evecs : the same format as in comp_pca.m
%
    EVecs = EVecs';
    for s = 1:10
       img(:,:,:,s) = reshape(EVecs(s,:), 28, 28,1)'; 
    end
    montage(img,'DisplayRange',[-0.2,0.2]);
end
