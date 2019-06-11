function my_mean = MyMean(X)
    [row col] = size(X);
    sum = zeros(1,col);
    for i = 1:row
       sum = sum(1,:) + X(i,:); 
    end
    my_mean = sum./row; 
end