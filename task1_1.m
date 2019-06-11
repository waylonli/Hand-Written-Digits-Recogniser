%
%
function task1_1(X, Y)
% Input:
%  X : M-by-D data matrix (double)
%  Y : M-by-1 label vector (unit8)
    [row col] = size(X);
    for k = 1:10
        j = 1;
        for i = 1:row
            if (Y(i,1) == (k-1))
                Xout(j,:) = X(i,:);
                j = j + 1;
            end
            if (j >= 11)
               break;
            end
        end
        for s = 1:10
            % Reshape it to 4D in order to use montage()
            img(:,:,:,s) = reshape(Xout(s,:), 28, 28,1)';
        end
        montage(img,'Size',[3 4])
        pause;
    end
end
