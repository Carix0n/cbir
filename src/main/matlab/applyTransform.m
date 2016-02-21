function [data] = applyTransform(data, U, descriptionVectorLength)
% [data] = applyTransform(data, U, descriptionVectorLength) applies PCA,
% whitening and re-normalization

    data = normalizeL2(data);
    data = projectData(data, U, descriptionVectorLength);
    data = normalizeL2(data);
    
end