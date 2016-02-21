function [neuralcodes] = readIBNC(ncFullNameList, descrVecLen, Nimages, maxPatchLevelQuery, GPU_MODE)
% [neuralcodes] = readIBNC(ncFullNameList, descrVecLen, Nimages,
% maxPatchLevelQuery, GPU_MODE) reads neural codes of images from the base
% 

    % Compute required values:
    patchesPerImageQuery = sum((1:maxPatchLevelQuery) .^ 2);
    
    % Open appropriate files with neural codes:
    fileNC = zeros(maxPatchLevelQuery, 1);
    for patchLevelQuery = 1:maxPatchLevelQuery
        fileNC(patchLevelQuery) = fopen(ncFullNameList{patchLevelQuery}, 'r');
    end
    
    % Allocate memory:
    neuralcodes = zeros(descrVecLen, Nimages * patchesPerImageQuery, 'single');
    if GPU_MODE
        neuralcodes = gpuArray(neuralcodes); 
    end
    
    % Read from files:
    for imageIndex = 1:Nimages
        headSum = 0;
        tailSum = 0;
        for patchLevelQuery = 1:maxPatchLevelQuery
            headSum = headSum + (patchLevelQuery - 1) ^ 2;
            tailSum = tailSum + patchLevelQuery ^ 2;
            head = (imageIndex - 1) * patchesPerImageQuery + headSum + 1;
            tail = (imageIndex - 1) * patchesPerImageQuery + tailSum;
            neuralcodes(:, head:tail) = single(fread(fileNC(patchLevelQuery), ...
                [descrVecLen, patchLevelQuery ^ 2], 'single'));
        end
    end

    % Close data files:
    for patchLevelQuery = 1:maxPatchLevelQuery
        fclose(fileNC(patchLevelQuery));
    end
    
end

