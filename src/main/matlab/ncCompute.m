function [] = ncCompute(imagebasePath, fdescription, net, ncFullName, patchLevel, GPU_MODE)
%
% [] = ncCompute(imagebasePath, fdescription, net, ncFullName, patchLevel, GPU_MODE)
%
% Computes neural codes for images from the base
%

    % Pre-trained nets:
    if GPU_MODE
        net = vl_simplenn_move(net, 'gpu');
        device = gpuDevice;
    else
        net = vl_simplenn_move(net, 'cpu');
        device = memory;
    end
    
    % Compute index of the first fully-connected layer:
    for layerIndex = 1:size(net.layers, 2)
        if ~isempty(strfind(net.layers{1, layerIndex}.name, 'fc'))
            firstFCIndex = layerIndex;
            break
        end
    end
    
    % Compute required values:
    patchesPerImage = patchLevel ^ 2;
    Nimages = length(fdescription);
    
    % Append path-prefix:
    for imageIndex = 1:Nimages
        fdescription(imageIndex).name = fullfile(imagebasePath, fdescription(imageIndex).name);
    end
    
    fileNC = fopen(ncFullName, 'w');
    
    rest = Nimages;
    while rest > 0
        requiredMemoryPerImage = 4 * patchesPerImage * (4096 + prod(net.normalization.imageSize));
        if GPU_MODE
            availableNum = device.AvailableMemory / requiredMemoryPerImage;
        else
            availableNum = device.MaxPossibleArrayBytes / requiredMemoryPerImage;
        end
        
        if availableNum > 1.1 * rest
            usedNum = rest;
        else
            usedNum = floor(0.9 * availableNum);
        end
        
        head = Nimages - rest + 1;
        tail = Nimages - rest + usedNum;
        neuralcodes = ncPatchCodes(fdescription(head:tail), net, patchLevel, firstFCIndex, GPU_MODE);
        
        if GPU_MODE
            neuralcodes = gather(neuralcodes);
        end
        
        fwrite(fileNC, neuralcodes, 'single');
        clear neuralcodes;
        
        rest = rest - usedNum;
    end
    
    fclose(fileNC);
    
end