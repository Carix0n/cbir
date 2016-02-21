%% ========================================================================
% Set names to directories and files
settingsDir = '..\..\..\config';
settingsFileName = fullfile(settingsDir, 'gist.conf');

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
wd = fullfile(rootDir, 'collections',  baseName);
imagebasePath = fullfile(wd, 'images');
fdescription = imdescription(imagebasePath);
gistPath = fullfile(wd, 'gist');
gistFileName = generateFileName({baseName; 'gist'}, '.bin');
gistFullName = fullfile(gistPath, gistFileName);
%% ========================================================================
% Compute appropriate data if has not been computed yet
if ~exist(gistPath, 'dir')
    mkdir(gistPath);
end

if ~exist(fullfile(gistPath, gistFileName), 'file')
    gistCompute(imagebasePath, fdescription, gistFullName);
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

resultFileName = generateFileName({baseName; imageShortenedName; 'gist'}, '.csv');
resultFullName = fullfile(resultPath, resultFileName);

gistFindNearest(inputImageFullName, imagebasePath, fdescription, gistFullName, resultFullName);