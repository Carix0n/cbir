function U = funPCA(X)
%PCA Run principal component analysis on the dataset X
%   U = funPCA(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U
%

    [~, m] = size(X);
    covarianceMatrix = (1 / m) * (X * X');
    [U, ~, ~] = svd(covarianceMatrix);
    
end