function [list] = getNetsNameList(path)
% [list] = getNetsNameList(path)
%
% Returns neurals networks namelist
%

    files = dir(path);
    files = files(~[files.isdir]);
    list(length(files)) = struct('name', []);
    for fileIndex = 1:length(files)
        fileName = files(fileIndex).name;
        list(fileIndex) = struct('name', fileName);
    end

end

