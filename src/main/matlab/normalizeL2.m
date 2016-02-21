function X = normalizeL2(X)
%
%   X = normalizeL2(X) normalizes vectors in columns of X
%   Returns the normalized data
%
    for col = 1:size(X, 2)
        current = X(:, col);
        X(:, col) = current / norm(current);
    end
    
end

