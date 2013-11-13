function [genY, genD, genW, genW0] = genData(genSizes, noiseLevel,... 
                                        sparsityCtrls, standardize)
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


if nargin < 1
  error('Dimensions are required as input');
end

assert(length(genSizes) == 3, 'genSizes must consist of {features, samples, atoms}');

features = genSizes{1};
samples = genSizes{2};
atoms = genSizes{3};

if isempty(noiseLevel)
    noiseLevel = 0;
end

if length(sparsityCtrls) < 1
    wSparsityPerc = -1;
else
    wSparsityPerc = sparsityCtrls{1}/100;
    if wSparsityPerc > 100
        wSparsityPerc = 1;
    end
end

if length(sparsityCtrls) < 2
    dictSparsityPerc = -1;
else
    dictSparsityPerc = sparsityCtrls{2}/100;
    if dictSparsityPerc > 100
        dictSparsityPerc = 1;
    end
end

%sparsityjiggle: Adds a bit of randomness to the wSparsitiPerc
%e.g. if sparsity for w is 60% and jiggle 10%, the sparsity can be
%between 50 and 70% for each column
if length(sparsityCtrls) < 3
    sj = 10;
else
    sj = sparsityCtrls{3};
end

if wSparsityPerc + sj/100 > 1
    sj = 1 - wSparsityPerc * 100;
end

if wSparsityPerc - sj/100 < 0
    sj = wSparsityPerc * 100;
end

%if isempty(standardize)
%    standardize = false;
%end

%% Generate W
genW = abs(randn(atoms, samples));
%in each column, sets a random number of columns 0
if wSparsityPerc ~= 0
    for i=1:size(genW, 2)
        %Decides how many rows will be set to 0
        if wSparsityPerc < 0
            r = randi(size(genW, 1) - 1);
        else
            jiggle = randi([-sj, sj])/100;
            r = round(size(genW, 1) * (wSparsityPerc + jiggle));
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
if dictSparsityPerc ~= 0
    for i=1:size(genD, 2)
        %Decides how many rows will be set to 0
        if dictSparsityPerc < 0
            r = randi([0, size(genD, 1) - 1]);
        else
            r = round(size(genD, 1) * dictSparsityPerc);
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
