function [] = showFindNearest(inputImageFile, imagebasePath, fdescription, sortedIndex)
% [] = showFindNearest(inputImageFile, imagebasePath, fdescription, sortedIndex)
%
% Shows top nearest images to the screen
%

    inputImage = imread(inputImageFile);
    figure;
    imshow(inputImage);
    title('Input image');
    Nrows = 2;
    Ncols = 5;
    figure;
    for row = 1:Nrows
        for col = 1:Ncols
            imageIndex = (row - 1) * Ncols + col;
            fileName = fullfile(imagebasePath, fdescription(sortedIndex(imageIndex)).name);
            Image = imread(fileName);
            subplot(Nrows, Ncols, imageIndex);
            imshow(Image);
            title(fdescription(sortedIndex(imageIndex)).name, 'Interpreter', 'none');
        end
    end
    
end

