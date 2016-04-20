function [] = ncFindNearest(inputImageFile, imagebasePath, fdescription, net, ibNC, U, resultFullName, maxPatchLevelRef, maxPatchLevelQuery, ...
    descrVecLen, GPU_MODE)
% [] = ncFindNearest(inputImageFile, imagebasePath, fdescription, net, ibNC, U, resultFullName, maxPatchLevelRef, maxPatchLevelQuery,
%   descrVecLen, GPU_MODE)
%
% Computes top-10 nearest images with neural codes
%

    % Pre-trained nets:
    if GPU_MODE
        net = vl_simplenn_move(net, 'gpu');
    else
        net = vl_simplenn_move(net, 'cpu');
    end
    
    % Compute index of the first fully-connected layer:
    for layerIndex = 1:size(net.layers, 2)
        if ~isempty(strfind(net.layers{layerIndex}.name, 'fc'))
            firstFCIndex = layerIndex;
            break
        end
    end
    
    % Compute required values:
    Nimages = length(fdescription);
    patchesPerImageQuery = sum((1:maxPatchLevelQuery) .^ 2);
    patchesPerImageRef = sum((1:maxPatchLevelRef) .^ 2);
    
    % Pre-allocation of input image patch codes:
    if GPU_MODE
        neuralcodesInput = zeros(4096, sum((1:maxPatchLevelRef) .^ 2), 'single', 'gpuArray');
    else
        neuralcodesInput = zeros(4096, sum((1:maxPatchLevelRef) .^ 2), 'single');
    end
    
    % Read image and compute neural codes:
    inputImage = single(imread(inputImageFile));
    netHeight = net.meta.normalization.imageSize(1);
    netWidth = net.meta.normalization.imageSize(2);
    inputImage = imresize(inputImage, [netHeight * (maxPatchLevelRef + 1) / 2, netWidth * (maxPatchLevelRef + 1) / 2]);
    for patchLevelRef = 1:maxPatchLevelRef
        head = sum((0:(patchLevelRef-1)) .^ 2) + 1;
        tail = sum((1:patchLevelRef) .^ 2);
        neuralcodesInput(:, head:tail) = ncPatchCodes(inputImage, net, patchLevelRef, firstFCIndex, GPU_MODE);
    end
    
    % Read neural codes from image database
    % and compute distances between images and sort:
    if GPU_MODE
        pairwiseDistTable = zeros(patchesPerImageRef, Nimages * patchesPerImageQuery, 'single', 'gpuArray');
        device = gpuDevice;
    else
        pairwiseDistTable = zeros(patchesPerImageRef, Nimages * patchesPerImageQuery, 'single');
        device = memory;
    end
    
    % If description vector's length < 4096, apply transform:
    if descrVecLen < 4096
        headSum = 0;
        tailSum = 0;
        if GPU_MODE
            neuralcodesInputPCA = zeros(descrVecLen, sum((1:maxPatchLevelQuery) .^ 2), 'single', 'gpuArray');
        else
            neuralcodesInputPCA = zeros(descrVecLen, sum((1:maxPatchLevelQuery) .^ 2), 'single');
        end
        for patchLevelRef = 1:maxPatchLevelRef
            headSum = headSum + (patchLevelRef - 1) ^ 2;
            tailSum = tailSum + patchLevelRef ^ 2;
            head = headSum + 1;
            tail = tailSum;
            
            neuralcodesInputPCA(:, head:tail) = applyTransform(neuralcodesInput(:, head:tail), U(:,:, patchLevelRef), descrVecLen);
        end
        neuralcodesInput = neuralcodesInputPCA;
        clear neuralcodesInputPCA;
        clear U;
        
    end

    totalNumCols = size(ibNC, 2);
    rest = totalNumCols;
    while rest > 0
        % 4 bytes * 4096 elements = 16384 bytes
        if GPU_MODE
            availableNumCols = device.AvailableMemory / 16384;
        else
            availableNumCols = device.MaxPossibleArrayBytes / 16384;
        end
        
        if availableNumCols > 2 * rest
            numCols = rest;
        else
            numCols = floor(0.5 * availableNumCols);
        end
        
        head = totalNumCols - rest + 1;
        tail = totalNumCols - rest + numCols;
        pairwiseDistTable(:, head:tail) = pairwiseDistance(neuralcodesInput, ibNC(:, head:tail));
        
        rest = rest - numCols;
    end
    
    pairwiseDistTable = reshape(pairwiseDistTable, [patchesPerImageRef, patchesPerImageQuery, Nimages]);
    minDistances = reshape(min(pairwiseDistTable, [], 2), [patchesPerImageRef, Nimages]);
    ncDif = mean(minDistances);
    
    % Ranking images:
    [~, sortedIndex] = sort(ncDif);
    
    % Define the number of nearest images:
    Nnearest = min(Nimages, 10);
    
    % Print nearest images to the resulting file:
    printFindNearest(resultFullName, Nnearest, fdescription, sortedIndex, ncDif);
    
    % Plot the sesult:
    showFindNearest(inputImageFile, imagebasePath, fdescription, sortedIndex);
    
end