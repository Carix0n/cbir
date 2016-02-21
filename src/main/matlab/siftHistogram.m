function [] = siftHistogram(centerFullName, clusterIndexFullNname, pointsnumFullName, ...
                            histFullName, fdescription)
%
% [] = siftHistogram(path, centerFileName, clusterIndexFileNname, pointsnumFileName, 
%                       histFileName, fdescription)
%
% Computes hostograms for SIFT-descriptors based on clusters on images from
% the imagebase
%
    
    % Read centers:
    fileClusterCenter = fopen(centerFullName, 'r');
    center = int32(fread(fileClusterCenter, [128, Inf], 'int32'));
    fclose(fileClusterCenter);
    
    % Computing histograms for images:
    Nimages = length(fdescription);
    Nclusters = size(center, 2);
    fileClusterIndex = fopen(clusterIndexFullNname, 'r');
    fileIBHist = fopen(histFullName, 'w');
    filePointsNum = fopen(pointsnumFullName, 'r');
    for imageIndex = 1:Nimages
        ibHist = zeros(Nclusters, 1);
        ibPointsNum = double(fread(filePointsNum, 1, 'uint32'));
        for pointIndex = 1:ibPointsNum
            clusterIndex = uint32(fread(fileClusterIndex, 1, 'uint32'));
            ibHist(clusterIndex) = ibHist(clusterIndex) + 1;
        end
        ibHist = ibHist / ibPointsNum;
        fwrite(fileIBHist, ibHist, 'double');
    end
    fclose(fileClusterIndex);
    fclose(fileIBHist);
    fclose(filePointsNum);
end