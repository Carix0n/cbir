function res = pairwiseDistance(X, Y)
% [res] = pairwiseDistance(X, Y)
% Computes pairwise distance matrix for vector in columns of X and Y, i.e.
% X is k-by-n matrix and Y is k-by-m matrix
% result matrix is n-by-m
%
    
    Xsize = size(X, 2);
    Ysize = size(Y, 2);
    
    % (x - y, x - y) = (x, x) - 2(x, y) + (y, y)
    X = X';
    X_norm = repmat(sum(X .^ 2, 2), 1, Ysize);
    Y_norm = repmat(sum(Y .^ 2), Xsize, 1);
    res = X_norm - 2 * X * Y + Y_norm;
    
end

