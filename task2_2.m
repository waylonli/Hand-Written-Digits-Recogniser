%
%
function Dmap = task2_2(Xtrain, Ytrain, k, MAT_evecs, MAT_evals, posVec, nbins)
% Input:
%  X   : M-by-D data matrix (double)
%  k   : scalar (integer) - the number of nearest neighbours
%  MAT_evecs : MAT filename of eigenvector matrix of D-by-D
%  MAT_evals : MAT filename of eigenvalue vector of D-by-1
%  posVec    : 1-by-D vector (double) to specity the position of the plane
%  nbins     : scalar (integer) - the number of bins for each PCA axis
% Output:
%  Dmap  : nbins-by-nbins matrix (uint8) - each element represents
%	   the cluster number that the point belongs to.

    load(MAT_evecs);
    load(MAT_evals);
    
    % The first two eigenvectors are two principal components
    principals = EVecs(:,1:2);
    [row col] = size(Xtrain);
    new_idx = zeros(1,1);
    % Projec posVec to 2D
    cen_plane = posVec * principals;
    
    % Get the half of width and height for the plane
    xrange = 5*sqrt(EVals(1,1));
    yrange = 5*sqrt(EVals(2,1));
    % Get the coordinate of the plane
    plane = cal_plane(xrange,yrange,nbins,cen_plane);
    
    % Extend the plane coordinate to 784D
    plane_D = zeros(40000,784);
    plane_D(:,1:2) = plane;
    
    % Move the position Xtrain and rotate it to fit the new axes
    Xtrain_new = zeros(row,col);
    for i = 1:row
        for j = 1:col
            Xtrain_new(i,j) = EVecs(:,j)' * (Xtrain(i,:) - posVec)' ;
        end
    end
     
    % Build the dmap matrix by applying knn classifier
    Dmap = zeros(nbins,nbins);
    Pred = run_knn_classifier(Xtrain_new, Ytrain, plane_D, k); 
    for j = 1:200
       Dmap(j,:) = Pred(((j-1)*200+1):j*200,1) ; 
    end
    Dmap = Dmap + 1;
    Dmap = uint8(Dmap);
     
%      save(['task2_2_Dmap_' num2str(k)],'Dmap');
     
    cmap = [0.80369089,  0.61814689,  0.46674357;
         0.81411766,  0.58274512,  0.54901962;
         0.58339103,  0.62000771,  0.79337179;
         0.83529413,  0.5584314 ,  0.77098041;
         0.77493273,  0.69831605,  0.54108421;
         0.72078433,  0.84784315,  0.30039217;
         0.96988851,  0.85064207,  0.19683199;
         0.93882353,  0.80156864,  0.4219608 ;
         0.83652442,  0.74771243,  0.61853136;
         0.7019608 ,  0.7019608 ,  0.7019608];
     
    row_min = min(plane(:,1)) - cen_plane(1,1);
    row_max = max(plane(:,1)) - cen_plane(1,1);
    col_min = min(plane(:,2)) - cen_plane(1,2);
    col_max = max(plane(:,2)) - cen_plane(1,2);
    % Visualise the decision regions
    [CC h] = contourf(linspace(row_min,row_max,200),linspace(col_min,col_max,200),Dmap);
    set(h,'LineColor','none');
    colormap(cmap);

end

function plane = cal_plane(xrange,yrange,nbins,cen_plane)
    % Calculate all the centres of those 40000 small squares
    plane = zeros(nbins*nbins,2);
    for i = 1:nbins*nbins
        row = ceil(i/200);
        col = i - 200 * (row-1);
        if col == 0
            col = 200;
        end
        x = cen_plane(1,1) - xrange + (2*(row-1)+1) * (xrange/nbins);
        y = cen_plane(1,2) - yrange + (2*(col-1)+1) * (yrange/nbins);
        plane(i,1) = x;
        plane(i,2) = y;
    end
end

