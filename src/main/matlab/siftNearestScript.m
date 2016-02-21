%% ========================================================================
% Set names to directories and files
settingsDir = '..\..\..\config';
settingsFileName = fullfile(settingsDir, 'sift.conf');

settingsFile = fopen(settingsFileName, 'r');
while 1
    command = fscanf(settingsFile, '%s', 3);
    if ~strcmp(command, '')
        eval(strcat(command, ';'))
    else
        break
    end
end
fclose(settingsFile);

rootDir; % Value storage in the settings file

baseName; % Value storage in the settings file

randSamplePercenatge; % Value storage in the settings file

Nclusters; % Value storage in the settings file

wd = fullfile(rootDir, 'collections', baseName);
imagebasePath = fullfile(wd, 'images');
fdescription = imdescription(imagebasePath);
dataPath = fullfile(wd, 'sift');
appendix = strjoin({num2str(randSamplePercenatge), num2str(Nclusters)}, '_');
siftFileName = generateFileName({baseName; 'sift'; num2str(randSamplePercenatge)}, '.bin');
siftFullName = fullfile(dataPath, siftFileName);
pointsnumFileName = generateFileName({baseName; 'pointsnum'; num2str(randSamplePercenatge)}, '.bin');
pointsnumFullName = fullfile(dataPath, pointsnumFileName);
centerFileName = generateFileName({baseName; 'center'; appendix}, '.bin');
centerFullName = fullfile(dataPath, centerFileName);
clusterIndexFileNname = generateFileName({baseName; 'clusterindex'; appendix}, '.bin');
clusterIndexFullNname = fullfile(dataPath, clusterIndexFileNname);
histFileName = generateFileName({baseName; 'histogram'; appendix}, '.bin');
histFullName = fullfile(dataPath, histFileName);
%% ========================================================================
% Compute appropriate data if has not been computed yet
if ~exist(dataPath, 'dir')
    mkdir(dataPath);
end

if ~exist(siftFullName, 'file') || ...
   ~exist(pointsnumFullName, 'file')
        fprintf('SIFT features with %f percentage of points are not available\nExtracting...\n', ...
            randSamplePercenatge);
        siftCompute(imagebasePath, fdescription, siftFullName,pointsnumFullName, randSamplePercenatge);
        fprintf('Completed\n');
end

if ~exist(centerFullName, 'file') || ~exist(clusterIndexFullNname, 'file')
    fprintf('Clusters with %f percentage of points and #clusters = %d are not available\nExtracting...\n', ...
        randSamplePercenatge, Nclusters);
    maxiters = 150;
    method = 'lloyd';
    siftClustering(siftFullName, centerFullName, clusterIndexFullNname, Nclusters, maxiters, method);
    fprintf('Completed\n');
end

if ~exist(histFullName, 'file')
    fprintf('Histograms with %f percentage of points and #clusters = %d are not available\nExtracting...\n', ...
            randSamplePercenatge, Nclusters);
    siftHistogram(centerFullName, clusterIndexFullNname, pointsnumFullName, histFullName, fdescription);
end
%% ========================================================================
% Set reference image and find nearest
testImageBaseDir = 'references';
inputImageDir = fullfile(rootDir, testImageBaseDir);
inputImageFileName; % Value storage in the settings file
inputImageFullName = fullfile(inputImageDir, inputImageFileName);

resultPath = fullfile(wd, 'result');

if ~exist(resultPath, 'dir')
    mkdir(resultPath);
end

[~, imageShortenedName, ~] = fileparts(inputImageFullName);

resultFileName = generateFileName({baseName; imageShortenedName; 'sift'}, '.csv');
resultFullName = fullfile(resultPath, resultFileName);

fprintf('Searching for nearest images...\n');
siftFindNearest(inputImageFullName, imagebasePath, fdescription, centerFullName, histFullName, ...
    resultFullName);