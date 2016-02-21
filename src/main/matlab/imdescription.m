function [fdescription] = imdescription(imagebasePath, randSamplePercenatge)
%
% [] = imdescription(ibdir)
% Writes the description of images for all ones from the input directory
%
    if nargin < 2 || randSamplePercenatge <= 0 || randSamplePercenatge > 1
        randSamplePercenatge = 1;
    end
    
    files = dir(imagebasePath);
    files = files(~[files.isdir]);
    if strcmp(files(1).name, 'Thumbs.db')
        files = files(2:end);
    end
    
    if randSamplePercenatge < 1
        Nimages = length(files);
        sampleIndices = randperm(Nimages, round(Nimages * randSamplePercenatge));
        files = files(sampleIndices);
    end
    
    % Create descriptions:
    fdescription = struct('name', {files.name});
end