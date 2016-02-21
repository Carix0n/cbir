function [] = siftClustering(siftFullName, centerFullName, clusterIndexFullName, ...
                                Nclusters, maxiters, method)
%
% [] = siftClustering(siftFullName, centerFullName, clusterIndexFullName, 
%                       Nclusters, maxiters, method)
%
% Computes clusters for SIFT-descriptors on images from the imagebase
% There are two available options for method: lloyd and elkan
% The lloyd method is default for undefined case and the lloyd method
% is faster
%
    
    % Working with algorithm's parametres:
    if nargin < 5
        Nclusters = 100;
    end
    if nargin < 6
        maxiters = 0;
    end
    if nargin < 7
        method = 'lloyd';
    end 
    if ~strcmp(method, 'lloyd') && ~strcmp(method, 'elkan')
        method = 'lloyd';
    end
    
    % Read SIFT-descriptor:
    fileSIFT = fopen(siftFullName, 'r');
    sift = uint8(fread(fileSIFT, [128, Inf], 'uint8'));
    fclose(fileSIFT);
    
    % Clustering
    % vl_ikmeans works with matrices where point is described as a column:
    if maxiters == 0
        [center, clusterIndex] = vl_ikmeans(sift, Nclusters, ...
            'method', method);
    else
        [center, clusterIndex] = vl_ikmeans(sift, Nclusters, ...
            'method', method, 'maxiters', maxiters);
    end
    
    % Write clustering results to the files:
    fileCenter = fopen(centerFullName, 'w');
    fwrite(fileCenter, center, 'int32');
    fclose(fileCenter);
    
    fileClusterIndex = fopen(clusterIndexFullName, 'w');
    fwrite(fileClusterIndex, clusterIndex, 'uint32');
    fclose(fileClusterIndex);
end