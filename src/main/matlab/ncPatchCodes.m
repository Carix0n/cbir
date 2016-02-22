function [neuralcodes] = ncPatchCodes(imbase, net, patchLevel, layerIndex, GPU_MODE)
% [neuralcodes] = ncPatchCodes(fdescription, net, patchLevel, layerIndex, GPU_MODE)
%
% Computes neural codes for patches of image on the PATCHLEVEL
%
    
    % Compute required values:
    patchesPerImage = patchLevel ^ 2;
    if isnumeric(imbase)
        Nimages = 1;
    else
        Nimages = length(imbase);
    end
    totalPathcesNum = Nimages * patchesPerImage;
    
    % Preallocation of ouput matrix:
    neuralcodes = zeros(4096, totalPathcesNum, 'single');
    patchArray = zeros([net.meta.normalization.imageSize, totalPathcesNum], 'single');
    if GPU_MODE
        neuralcodes = gpuArray(neuralcodes);
        patchArray = gpuArray(patchArray);
    end
    
    curIndex = 0;
    for imageIndex = 1:Nimages
        if ~isnumeric(imbase)
            fileName = imbase(imageIndex).name;
            im = single(imread(fileName));
        else
            im = imbase;
            clear imbase;
        end
        netHeight = net.meta.normalization.imageSize(1);
        netWidth = net.meta.normalization.imageSize(2);
        im = imresize(im, [netHeight * (patchLevel + 1) / 2, netWidth * (patchLevel + 1) / 2]);
        imHeight = size(im, 1);
        imWidth = size(im, 2);
        for patchVerIndex = 1:patchLevel
            verBound1 = round((patchVerIndex - 1) / (patchLevel + 1) * imHeight) + 1;
            verBound2 = round((patchVerIndex + 1) / (patchLevel + 1) * imHeight);
            for patchHorIndex = 1:patchLevel
                horBound1 = round((patchHorIndex - 1) / (patchLevel + 1) * imWidth) + 1;
                horBound2 = round((patchHorIndex + 1) / (patchLevel + 1) * imWidth);
                patch = im(verBound1:verBound2, horBound1:horBound2, :);
                curIndex = curIndex + 1;
                if GPU_MODE
                    patchArray(:,:,:, curIndex) = gpuArray(patch);
                else
                    patchArray(:,:,:, curIndex) = patch;
                end
            end
        end
    end
    
    patchArray = bsxfun(@minus, patchArray, net.meta.normalization.averageImage);
    
    for patchIndex = 1:totalPathcesNum
        nnoutputs = vl_simplenn(net, patchArray(:,:,:, patchIndex));
        code = nnoutputs(layerIndex + 1).x;
        neuralcodes(:, patchIndex) = code(:);
    end
    
end