function [param] = setGISTParam()
% [params] = setGISTParams()
% Set GIST params to the appropriate

    % set a normalized image size
    param.imageSize = [256 256];
    % number of orientations per scale (from HF to LF)
    param.orientationsPerScale = [8 8 8 8];
    param.numberBlocks = 4;
    param.fc_prefilt = 4;
    
end