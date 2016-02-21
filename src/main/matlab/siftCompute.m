function [] = siftCompute(imagebasePath, fdescription, siftFullName, ...
                            pointsnumFullName, randSamplePercenatge)
%
% [] = siftCompute(imagebasePath, fdescription, siftFullName, 
%                   pointsnumFullName, randSamplePercenatge)
%
% Computes SIFT-descriptors for images from the base
%
    if nargin < 5 || randSamplePercenatge <= 0 || randSamplePercenatge > 1
        randSamplePercenatge = 1;
    end
    
    % Compute SIFT and write to the file:
    Nimages = length(fdescription);
    fileSIFT = fopen(siftFullName, 'w');
    ibPointsNum = uint32(zeros(Nimages, 1));
    for imageIndex = 1:Nimages
        imgFileName = fdescription(imageIndex).name;
        img = single(rgb2gray(imread(fullfile(imagebasePath, imgFileName))));
        [~, sift] = vl_sift(img);
        if randSamplePercenatge < 1
            siftSize = size(sift, 2);
            sampleIndices = randperm(siftSize, round(siftSize * randSamplePercenatge));
            sift = sift(:, sampleIndices);
        end
        fwrite(fileSIFT, sift, 'uint8');
        ibPointsNum(imageIndex) = size(sift, 2);
    end
    fclose(fileSIFT);
    
    % Write number of points for every image to the file:
    filePointsNum = fopen(pointsnumFullName, 'w');
    fwrite(filePointsNum, ibPointsNum, 'uint32');
    fclose(filePointsNum);
end