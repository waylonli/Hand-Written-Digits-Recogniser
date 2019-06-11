function sq_dist = MySqDist(U, v)
    sq_dist = sum(bsxfun(@minus, U, v).^2, 2)';
end