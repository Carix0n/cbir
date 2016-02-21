function [X] = featureRecover(X_norm, mu, sigma)
%FEATURERECOVER Recovers the features in X_norm 
%   FEATURERECOVER(X) returns an original version of X.

    X = X_norm;
    
    X = bsxfun(@times, X, sigma);

    X = bsxfun(@plus, X, mu);

end
