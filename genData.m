function [genY, genD, genW, genW0] = genData(features, samples, atoms, noiseLevel, wSparsityLevel, dictSparsityLevel)
% genData  generates synthetic data
%   [Y, D] = ADDME(f, s, a) generates Y of size f*s, with no noise added 
%   and D of size f*a
%
%   Additional parameters:
%   noiseLevel n: 0<= n <= 1 decides how much of the randomly generated
%                      noise will be added to Y
%
%   wSparsityLevel as well as dictSparsityLevel n:
%   If n = -1: Sets a random number of rows, in each column to 0 (for W or the
%   Dictionary respectively)
%   If 0<= n <= 1: Sets the given percentage of rows to 0. Which rows
%   though, are chosen randomly per column


if nargin < 3
  error('Dimensions are required as input')
end

if isempty(noiseLevel)
    noiseLevel = 0;
end

if isempty(wSparsityLevel)
    wSparsityLevel = -1;
else
    if wSparsityLevel > 1
        wSparsityLevel = 1;
    end
end

if isempty(dictSparsityLevel)
    dictSparsityLevel = -1;
else
    if dictSparsityLevel > 1
        dictSparsityLevel = 1;
    end
end
%% Generate W
genW = randn(atoms, samples);
%in each column, sets a random number of columns 0
if wSparsityLevel ~= 0
    for i=1:size(genW, 2)
        %Decides how many rows will be set to 0
        if wSparsityLevel < 0
            r = randi(size(genW, 1) - 1);
        else
            r = round(size(genW, 1) * wSparsityLevel);
        end
        %sets the first r rows to 0
        genW(1:r, i) = 0;
        %shuffles the whole row
        genW(:, i) = genW(randperm(length(genW(:, i))), i);
        %genW = genW .* (genW > 0);
    end
end
genW0 = zeros(1, samples);

%% Generate Dictonary (random normal dist. generation)
genD = abs(randn(features, atoms));
%in each column, sets a random number of rows to 0 (between 0 and "all-1")
if dictSparsityLevel ~= 0
    for i=1:size(genD, 2)
        %Decides how many rows will be set to 0
        if dictSparsityLevel < 0
            r = randi([0, size(genD, 1) - 1]);
        else
            r = round(size(genD, 1) * dictSparsityLevel);
        end
        %sets the first r rows to 0
        genD(1:r, i) = 0;
        %shuffles the whole column
        genD(:, i) = genD(randperm(length(genD(:, i))), i);
    end
end

%% Generate Y   
genY = predictY(genD, genW, genW0) + noiseLevel * abs(randn(features, samples));
end
