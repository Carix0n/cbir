%% ========================================================================
% Set names to directories and files
settingsDir = fullfile('..', '..', '..', 'config');
settingsFileName = fullfile(settingsDir, 'nc.conf');

settingsFile = fopen(settingsFileName, 'r');
while 1
    command = fscanf(settingsFile, '%s', 3);
    if ~isempty(command)
        eval(strcat(command, ';'))
    else
        break
    end
end
fclose(settingsFile);

rootDir; % Value storage in the settings file

baseName; % Value storage in the settings file

wd = fullfile(rootDir, 'collections', baseName);

imagebasePath = fullfile(wd, 'images');

fdescription = imdescription(imagebasePath);

ncPath = fullfile(wd, 'neuralcodes');

netsPath = fullfile(rootDir, 'nets');
netFileName; % Value storage in the settings file
netFullName = fullfile(netsPath, netFileName);
net = load(netFullName);
net = vl_simplenn_tidy(net); % Update pre-trained model if used the old version
[~, netName, ~] = fileparts(netFullName);

maxPatchLevelQuery; % Value storage in the settings file
maxPatchLevelRef; % Value storage in the settings file
maxPatchLevel = max(maxPatchLevelQuery, maxPatchLevelRef);

ncFullNameList = cell(maxPatchLevel, 1);
ncFullNamePCAList = cell(maxPatchLevel, 1);
ncFullNameMatrixPCAList = cell(maxPatchLevel, 1);

descrVecLen; % Value storage in the settings file

GPU_MODE; % Value storage in the settings file

% Compute required values:
Nimages = length(fdescription);
patchesPerImageQuery = sum((1:maxPatchLevelQuery) .^ 2);

for patchLevel = 1:maxPatchLevel
    ncFileName = generateFileName({baseName; netName; 'nc'; int2str(patchLevel)}, '.bin');
    ncFileNamePCA = generateFileName({baseName; netName; 'nc'; int2str(patchLevel); int2str(descrVecLen)}, '.bin');
    ncFileNameMatrixPCA = generateFileName({baseName; netName; 'pca'; int2str(patchLevel)}, '.bin');
    ncFullNameList{patchLevel} = fullfile(ncPath, ncFileName);
    ncFullNamePCAList{patchLevel} = fullfile(ncPath, ncFileNamePCA);
    ncFullNameMatrixPCAList{patchLevel} = fullfile(ncPath, ncFileNameMatrixPCA);
end
%% ========================================================================
% Compute appropriate data if has not been computed yet
if ~exist(ncPath, 'dir')
    mkdir(ncPath);
end

for patchLevel = 1:maxPatchLevel
    ncFullName = ncFullNameList{patchLevel};
    if ~exist(ncFullName, 'file')
        fprintf('Features from %s network on patch level #%d are not available\nExtracting...\n', netName, patchLevel);
        
        ncCompute(imagebasePath, fdescription, net, ncFullName, patchLevel, GPU_MODE);
        
        fprintf('Completed\n');
    end
    
    ncFullNameMatrixPCA = ncFullNameMatrixPCAList{patchLevel};
    if descrVecLen < 4096 && ~exist(ncFullNameMatrixPCA, 'file')
        fprintf('PCA matrix on level #%d is not available\nComputing...\n', patchLevel);
        
        ncFile = fopen(ncFullName, 'r');
        neuralcodes = single(fread(ncFile, [4096, inf], 'single'));
        fclose(ncFile);
        
        neuralcodes = normalizeL2(neuralcodes);
        U = funPCA(neuralcodes);
        
        ncFileMatrixPCA = fopen(ncFullNameMatrixPCA, 'w');
        fwrite(ncFileMatrixPCA, U, 'single');
        fclose(ncFileMatrixPCA);
        
        clear U;
        
        fprintf('Completed\n');
    end
    
    ncFullNamePCA = ncFullNamePCAList{patchLevel};
    if descrVecLen < 4096 && ~exist(ncFullNamePCA, 'file')
        fprintf('PCA features with #dim = %d from %s network on patch level #%d are not available\nExtracting...\n',...
            descrVecLen, netName, patchLevel);

        ncFile = fopen(ncFullName, 'r');
        neuralcodes = single(fread(ncFile, [4096, inf], 'single'));
        fclose(ncFile);
        
        ncFileMatrixPCA = fopen(ncFullNameMatrixPCA, 'r');
        U = single(fread(ncFileMatrixPCA, [4096, 4096], 'single'));
        fclose(ncFileMatrixPCA);
        
        neuralcodes = applyTransform(neuralcodes, U, descrVecLen);
        
        ncFilePCA = fopen(ncFullNamePCA, 'w');
        fwrite(ncFilePCA, neuralcodes, 'single');
        fclose(ncFilePCA);
        
        clear neuralcodes;
        clear U;

        fprintf('Completed\n');
    end
end
%% ========================================================================
resultPath = fullfile(wd, 'result');

if ~exist(resultPath, 'dir')
    mkdir(resultPath);
end

% Set reference image and find nearest
testImageBaseDir = 'references';
inputImageDir = fullfile(rootDir, testImageBaseDir);
inputImageFileName; % Value storage in the settings file
inputImageFullName = fullfile(inputImageDir, inputImageFileName);

[~, imageShortenedName, ~] = fileparts(inputImageFullName);

resultFileName = generateFileName({baseName; imageShortenedName; 'nc'}, '.csv');
resultFullName = fullfile(resultPath, resultFileName);

% Open appropriate files with neural codes:
if descrVecLen < 4096
    ncFullNameList = ncFullNamePCAList;
end

modelSizeIBNC = [descrVecLen, Nimages * patchesPerImageQuery];

if ~exist('ibNC', 'var') || ~isequal(size(ibNC), modelSizeIBNC)
    ibNC = readIBNC(ncFullNameList, descrVecLen, Nimages, maxPatchLevelQuery, GPU_MODE);
elseif GPU_MODE
    ibNC = gpuArray(ibNC);
else
    ibNC = gather(ibNC);
end

modelSizeU = [4096, 4096, maxPatchLevel];

if ~exist('U', 'var') || ~isequal(size(U), modelSizeU)
    U = readMatrixPCA(ncFullNameMatrixPCAList, maxPatchLevelRef, GPU_MODE);
elseif GPU_MODE
    U = gpuArray(U);
else
    U = gather(U);
end

fprintf('Searching for nearest images...\n');
ncFindNearest(inputImageFullName, imagebasePath, fdescription, net, ibNC, U, resultFullName, maxPatchLevelRef, maxPatchLevelQuery, descrVecLen, ...
    GPU_MODE);
fprintf('Done!\n');