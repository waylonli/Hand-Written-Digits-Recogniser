%
%
function Dmap = task1_7(MAT_ClusterCentres, MAT_M, MAT_evecs, MAT_evals, posVec, nbins)
% Input:
%  MAT_ClusterCentres: MAT filename of cluster centre matrix
%  MAT_M     : MAT filename of mean vectors of (K+1)-by-D, where K is
%              the number of classes (which is 10 for the MNIST data)
%  MAT_evecs : MAT filename of eigenvector matrix of D-by-D
%  MAT_evals : MAT filename of eigenvalue vector of D-by-1
%  posVec    : 1-by-D vector (double) to specify the position of the plane
%  nbins     : scalar (integer) to specify the number of bins for each PCA axis
% Output
%  Dmap  : nbins-by-nbins matrix (uint8) - each element represents
%	   the cluster number that the point belongs to.
    load(MAT_ClusterCentres);
    load(MAT_evecs);
    load(MAT_evals);
    load(MAT_M);
    
    % The first two eigenvectors are two principal components
    principals = EVecs(:,1:2);
    % Project the centres to 2D
    point = C * principals;

    [row col] = size(point);

    cen_plane = posVec * principals;
    
    % Get the half of width and height for the plane
    xrange = 5*sqrt(EVals(1,1));
    yrange = 5*sqrt(EVals(2,1));
    
    % Build the dmap matrix by using my classify function
    Dmap = zeros(nbins,nbins);
    for i = 1:nbins
        for j = 1:nbins
            Dmap(i,j) = classify(cen_plane,i,j,point,xrange,yrange,nbins);
        end
    end
    Dmap = uint8(Dmap);
%    save(['task1_7_dmap_' num2str(length(C(:,1)))],'Dmap');
    
    plane = cal_plane(xrange,yrange,nbins,cen_plane);
    % Initialise the colours
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
    % Get the coordinate range of the plane
    row_min = min(plane(:,1)) - cen_plane(1,1);
    row_max = max(plane(:,1)) - cen_plane(1,1);
    col_min = min(plane(:,2)) - cen_plane(1,2);
    col_max = max(plane(:,2)) - cen_plane(1,2);
    
    if length(C(:,1)) == 1
        % Contourf cannot be used when k = 1, so I use fill() instead
        fill([row_min row_max row_max row_min],[col_min col_min col_max col_max],cmap(1,:));
    else
        [CC h] = contourf(linspace(row_min,row_max,200),linspace(col_min,col_max,200),Dmap);
        set(h,'LineColor','none');
        colormap(cmap);
    end
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

function class = classify(cen_plane,row,col,point_in_plane,xrange,yrange,nbins)
    % This function is calculating the centres of the small squares
    % It also calculates the distance between the small squares and the centres
    % Finally classify them by finding the shortest distance
    x = cen_plane(1,1) - xrange + (2*(row-1)+1) * (xrange/nbins);
    y = cen_plane(1,1) - yrange + (2*(col-1)+1) * (yrange/nbins);
    [rows, ~] = size(point_in_plane);
    for i = 1:rows
        distance = square_dist([x y],point_in_plane(i,:));
        if i == 1
            min = distance;
            class = 1;
        elseif distance < min
            min = distance;
            class = i;
        end
    end
end

function sq_dist = square_dist(U, v)
    sq_dist = sum(bsxfun(@minus, U, v).^2, 2)';
end
