function [] = siftFindNearest(inputImageFile, imagebasePath, fdescription, ...
    centerFullName, histFullName, resultFullName, randSamplePercenatge)
%
% [] = siftFindNearest(inputImageFile, imagebasePath, fdescription, 
%           centerFullName, histFullName, resultFullName, randSamplePercenatge)
% Computes top nearest images with SIFT-descriptor
%
    if nargin < 8 || randSamplePercenatge <= 0 || randSamplePercenatge > 1
        randSamplePercenatge = 1;
    end
    
    % Read coordinates of clusters' centers:
    fileClusterCenter = fopen(centerFullName, 'r');
    center = int32(fread(fileClusterCenter, [128, Inf], 'int32'));
    fclose(fileClusterCenter);
    
    % Compute number of clusters:
    Nclusters = size(center, 2);
    
    % Read image and compute SIFT:
    inputImage = imread(inputImageFile);
    [~, inputSIFT] = vl_sift(single(rgb2gray(inputImage)));
    if randSamplePercenatge < 1
        siftSize = size(inputSIFT, 2);
        sampleIndices = randperm(siftSize, round(siftSize * randSamplePercenatge));
        inputSIFT = inputSIFT(:, sampleIndices);
    end
    inputSIFT = int32(inputSIFT);
    inputImagePointsNum = size(inputSIFT, 2);
    
    % Compute histogram of input image:
    pairwiseDistMatrix = pairwiseDistance(single(inputSIFT), single(center));
    [~, minDistanceIndex] = min(pairwiseDistMatrix, [], 2);
    inputImageHist = zeros(Nclusters, 1);
    for pointIndex = 1:inputImagePointsNum
        nearest = minDistanceIndex(pointIndex);
        inputImageHist(nearest) = inputImageHist(nearest) + 1;
    end
    inputImageHist = inputImageHist / inputImagePointsNum;
    
    % Compute distances between histograms:
    Nimages = length(fdescription);
    fileHistogram = fopen(histFullName, 'r');
    ibHist = double(fread(fileHistogram, [Nclusters Nimages], 'double'));
    fclose(fileHistogram);
    distToHist = histDist(inputImageHist, ibHist);
    [~, sortedIndex] = sort(distToHist);
    
    % Define the number of nearest images:
    Nnearest = min(Nimages, 10);
    
    % Print nearest images to the resulting file:
    printFindNearest(resultFullName, Nnearest, fdescription, sortedIndex, distToHist);
    
    % Plot the sesult:
    showFindNearest(inputImageFile, imagebasePath, fdescription, sortedIndex);
    
end