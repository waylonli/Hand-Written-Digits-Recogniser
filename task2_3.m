%
%
function task2_3(X, Y)
% Input:
%  X : M-by-D data matrix (double)
%  Y : M-by-1 label vector for X (unit8)


    [EVecs, EVals] = comp_pca(X);
    principals = EVecs(:,1:2);
    % Project the vectors to 2D
    X_2D = X * principals;

 
    for num = 1:10
        % Pick out the vectors with the same label
        X_num = X_2D(Y == (num-1),:);
        x_mean = zeros(1,2);
        Cov = zeros(2,2);
        for i = 1:2
            a = sum(X_num(:,i));
            b = length(X_num(:,1));
            x_mean(1,i) = a / b;
        end
        % Calculate the covariance matrix for each class
        for i = 1:size(X_num(:,1))
            Cov = Cov + ((X_num(i,:) - x_mean)')*(X_num(i,:) - x_mean)/b;
        end
        
        % Draw the contour of Gaussian distribution
        contourGauss2D(x_mean, Cov);
        text(x_mean(1,1),x_mean(1,2),num2str(num-1));
        hold on;
    end

end

function contourGauss2D(mu, covar)
     step = 0.1;
     C = covar; 
     A = inv(C); 
     var = diag(C);
    
     % Calculate the eigenvector and eigenvalue for the covariance matrix of each class
     [evec eval] = eig(C);
     eval = diag(eval);
     % Pick out the greater eigenvalue and the eigenvector
     if eval(1,1) >= eval(2,1)
         eval_L = eval(1,1);
         evec_L = evec(:,1)';
         eval_S = eval(2,1);
         evec_S = evec(:,2)';
     else
         eval_S = eval(1,1);
         evec_S = evec(:,1)';
         eval_L = eval(2,1);
         evec_L = evec(:,2)';
     end
     % Calculate the angle
     a = atan(evec_L(1)/evec_L(2));
    
     % Calculate the parameters of the ellipse
     x = sqrt(eval_L) * cos(linspace(0,2*pi));
     y = sqrt(eval_S) * sin(linspace(0,2*pi));
     % Construct the rotated matrix
     r = [cos(a) sin(a);
         -sin(a) cos(a)];
     temp = r * [y;x] ;
     % Plot the ellipse
     plot(temp(1,:) + mu(:,1), temp(2,:) + mu(:,2));        
end