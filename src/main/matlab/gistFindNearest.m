function [] = gistFindNearest(inputImageFile, imagebasePath, fdescription, ...
                                gistFullName, resultFullName)
%
% [] = gistFindNearest(inputImageFile, imagebasePath, fdescription, gistFullName)
% Computes top-10 nearest images with GIST-descriptor
%

    % GIST Parameters:
    param = setGISTParam();
    
    % Compute number of GIST-features:
    Nfeatures = sum(param.orientationsPerScale)*param.numberBlocks^2;

    % Read image and compute GIST:
    inputImage = imread(inputImageFile);
    inputGIST = LMgist(inputImage, '', param);
    inputGIST = inputGIST';
    
    % Read GIST values from image database
    % and compute distances between images and sort:
    fileGIST = fopen(gistFullName, 'r');
    Nimages = length(fdescription);
    ibGIST = single(fread(fileGIST, [Nfeatures Nimages], 'single'));
    fclose(fileGIST);
    gistDif = pairwiseDistance(ibGIST, inputGIST);
    [~, sortedIndex] = sort(gistDif);
    
    % Define the number of nearest images:
    Nnearest = min(Nimages, 10);
    
    % Print nearest images to the resulting file:
    printFindNearest(resultFullName, Nnearest, fdescription, ...
        sortedIndex, gistDif);
    
    % Plot the sesult:
    showFindNearest(inputImageFile, imagebasePath, fdescription, sortedIndex);
    
end