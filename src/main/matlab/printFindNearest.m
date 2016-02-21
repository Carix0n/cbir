function [] = printFindNearest(resultFullName, Nnearest, fdescription, sortedIndex, distArray)
%
% [] = printFindNearest(resultFullName Nnearest, fileName, fdescription, sortedIndex, distArray)
% Writes top nearest images to the resulting file
%

    output = fopen(resultFullName, 'w');
    formatSpec = '%s;%2.8f;\n';
    for i = 1:Nnearest
        imageFileName = fdescription(sortedIndex(i)).name;
        distOut = distArray(sortedIndex(i));
        fprintf(output, formatSpec, imageFileName, distOut);
    end
    fclose(output);
    
end

