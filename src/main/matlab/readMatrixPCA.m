function [U] = readMatrixPCA(ncFullNameMatrixPCAList, maxPatchLevelRef, GPU_MODE)
% [U] = readMatrixPCA(ncFullNameMatrixPCAList, maxPatchLevelRef, GPU_MODE)
% reads PCA matrices from the files
%
    % Allocate memory:
    U = zeros(4096, 4096, maxPatchLevelRef, 'single');
    if GPU_MODE
        U = gpuArray(U);   
    end
    
    % Read PCA matrices:
    for patchLevelRef = 1:maxPatchLevelRef
        ncFullNameMatrixPCA = ncFullNameMatrixPCAList{patchLevelRef};
        ncFileMatrixPCA = fopen(ncFullNameMatrixPCA, 'r');
        U(:,:, patchLevelRef) = single(fread(ncFileMatrixPCA, [4096, 4096], 'single'));
        fclose(ncFileMatrixPCA);
    end
    
end

