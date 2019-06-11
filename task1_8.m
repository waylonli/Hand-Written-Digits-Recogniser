%
%
function task1_8(X,Ks)
%  NB: there is no specification to this function.

    for i = 1:length(Ks)
        % Generate three groups of initial centres by using different methods
        ini_first10 = X(1:Ks(1,i),:);
        ini_random = pick_randomly(X,Ks(1,i));
        ini_far = pick_far(X,Ks(1,i));
        
        for j = 1:5
            % Get three groups of SEE value by applying kmeans clustering to different initial centres
            [~, ~, SSE_fst] = my_kMeansClustering(X, Ks(1,i), ini_first10);
            [~, ~, SSE_r] = my_kMeansClustering(X, Ks(1,i), ini_random);
            [~, ~, SSE_fa] = my_kMeansClustering(X, Ks(1,i), ini_far);
            if j == 1
                SSE_first10 = SSE_fst;
                SSE_random = SSE_r;
                SSE_far = SSE_fa;
            else
                SSE_first10 = SSE_first10 + SSE_fst;
                SSE_random = SSE_random + SSE_r;
                SSE_far = SSE_far + SSE_fa;
            end
        end
        
        % Plot the figure of three groups of initial centres
        plot(1:length(SSE_first10),SSE_first10(:,1)./5);
        hold on;
        plot(1:length(SSE_random),SSE_random(:,1)./5);
        hold on;
        plot(1:length(SSE_far),SSE_far(:,1)./5);
        hold on;
        title(['SSE for k = ' num2str(Ks(1,i))]);
        xlabel('Iteration number');
        ylabel('SSE');
        legend('First ten','Random','Far');
    end
end

function ini_random = pick_randomly(X,k)
    % Randomly pick k vectors as initial centres
    r = randperm(length(X(:,1)));
    ini_random(1:k,:) = [X(r(1,1:k),:)];
end

function ini_far = pick_far(X,k)
    % Always choose the vector which is furthest to the other centres we already had
    % Randomly pick the first centre
   r = randperm(length(X(:,1)));
   ini_far = zeros(k,length(X(1,:)));
   ini_far(1,:) = X(r(1,1),:);
   
   % Delete the vector which we already put as the initial centre
   temp1 = X(1:r-1,:);
   temp2 = X(r+1:length(X(:,1)),:);
   X = zeros(length(X(:,1))-1,length(X(1,:)));
   X = [temp1; temp2];
   if k >= 2
      for i = 2:k         
         distance = zeros(1,length(X(:,1)));
         for j = 1:i-1
            % Calculate the distance between the centres and the other vectors
            distance = distance + square_dist(ini_far(j,:),X);
         end
         
         % Get the index of the vector which is the furthest
         [~, idx] = max(distance);
         ini_far(i,:) = X(idx,:);
         % Delete the vector we put as one of the initial centres
         temp1 = X(1:r-1,:);
         temp2 = X(r+1:length(X(:,1)),:);
         X = zeros(length(X(:,1))-1,length(X(1,:)));
         X = [temp1; temp2];
      end
   end
end

function sq_dist = square_dist(U, v)
    sq_dist = sum(bsxfun(@minus, U, v).^2, 2)';
end