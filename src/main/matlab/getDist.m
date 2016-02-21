function [res] = getDist(x, y, metrics)
%
% [res] = getDist(x, y, metrics)
% Computes distance between x and y with some metrics
%
    % Transp if it is a case of row-vector and column-vector
    if size(x, 1) == size(y, 2) && size(x, 2) == size(y, 1)
        y = y';
    end
    
    switch metrics
        case 'L2'
            res = sum((x - y).^2);
        case 'hist'
            res = 1 - sum(min(x, y));
    end
end