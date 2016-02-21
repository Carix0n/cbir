function [res] = histDist(X, Y)
% [distMatrix] = histDist(X, Y)
%
% Computes histogram distance matrix for vector X to vector-columns in Y
%
    Y_size = size(Y, 2);
    X = repmat(X, 1, Y_size);
    res = ones(1, Y_size) - sum(min(X, Y));
    
end

