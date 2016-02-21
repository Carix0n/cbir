%% ========================================================================
% Set names to directories and files
settingsDir = '..\..\..\config';
settingsFileName = fullfile(settingsDir, 'nc.conf');

settingsFile = fopen(settingsFileName, 'r');
params = {'rootDir', 'maxPatchLevelQuery', 'maxPatchLevelRef', 'GPU_MODE'};
while 1
    variable = fscanf(settingsFile, '%s');
    value = fscanf(settingsFile, '%s', 2);
    if ~isempty(variable)
        if ~isempty(find(strcmp(variable, params), 1))
            eval(strcat(variable, value, ';'))
        end
    else
        break
    end
end
fclose(settingsFile);

netsPath = fullfile(rootDir, 'nets');
baseNameList = {'robots.ox.ac.uk'; 'lear.inrialpes.fr'; 'spatial_envelope'};
netsNameList = getNetsNameList(netsPath);

maxPatchLevel = max(maxPatchLevelQuery, maxPatchLevelRef);
for baseIndex = 1:length(baseNameList)
    baseName = baseNameList{baseIndex, 1};
    fprintf('Working with %s imagebase\n', baseName);
    wd = fullfile(rootDir, baseName);
    imagebasePath = fullfile(wd, 'images');
    fdescription = imdescription(imagebasePath);
    ncPath = fullfile(wd, 'neuralcodes');
    for netIndex = 1:length(netsNameList)
        netFileName = netsNameList(netIndex).name;
        netFullName = fullfile(netsPath, netFileName);
        [~, netName, ~] = fileparts(netFullName);
        fprintf('\tUsing %s net\n', netName);
        net = load(netFullName);
        if GPU_MODE
            net = vl_simplenn_move(net, 'gpu');
        end
        for patchLevel = 1:maxPatchLevel
            fprintf('\t\tPatch level = %d\n', patchLevel);
            ncFileName = generateFileName({baseName; netName; 'nc'; int2str(patchLevel)}, '.bin');
            ncFullName = fullfile(ncPath, ncFileName);
            if ~exist(ncPath, 'dir')
                mkdir(ncPath);
            end
            if ~exist(ncFullName, 'file')
                ncCompute(imagebasePath, fdescription, net, ncFullName, patchLevel, GPU_MODE);
            end
        end
    end
end