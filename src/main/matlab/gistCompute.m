function [] = gistCompute(imagebasePath, fdescription, gistFullName)
%
% [] = gistCompute(imagebaseDirPath, fdescription, gistPath, gistFileName)
%
% Computes GIST-descriptors for images from the base
%

    % GIST Parameters:
    param = setGISTParam();
    
    % Load first image, compute gist and write to the file:
    Nimages = length(fdescription);
    fileGIST = fopen(gistFullName, 'w');
    imgFileName = fdescription(1).name;
    img = imread(fullfile(imagebasePath, imgFileName));
    [gist, param] = LMgist(img, '', param); % first call
    fwrite(fileGIST, gist, 'single');
    % Loop:
    for imageIndex = 2:Nimages
        imgFileName = fdescription(imageIndex).name;
        img = imread(fullfile(imagebasePath, imgFileName));
        gist = LMgist(img, '', param); % the next calls will be faster
        fwrite(fileGIST, gist, 'single');
    end
    fclose(fileGIST);
end