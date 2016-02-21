function [fileName] = generateFileName(components, extension)
% [] = generateFileName(components, extension)
% Generates filename from the components and extension

    if isempty(components)
        return
    end
    
    fileName = strcat(strjoin(components, '_'), extension);
    
end

