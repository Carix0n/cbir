function [] = loadPointsDescription(descrDirStr)
%
% [] = loadPointsNum(descrDirStr)
% Writes number of SIFT-points for images:
%
    % Read the list of description files:
    files = dir(descrDirStr);
    files = files(~[files.isdir]);

    % Loop through the description files:
    len = length(files);
    ibPointsNum = zeros(len, 1);
    for fileIndex = 1:len
        fullFilePath = fullfile(descrDirStr, files(fileIndex).name);
        descrFile = fopen(fullFilePath, 'r');
        tmp = fscanf(descrFile, '%f\n%d\n', [2 1]);
        ibPointsNum(fileIndex) = tmp(2);
        descr = fscanf(descrFile, '%d %f %f %f %f %f\n', [6, tmp(2)]);
        fclose(descrFile);
    end
end