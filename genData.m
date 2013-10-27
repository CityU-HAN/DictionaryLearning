function [genY, genD] = genData(features, samples, atoms)
%% Generate w: All values bigger than 0 will be 1, all smaller 0
genW = randn(atoms, samples);
genW = genW .* (genW > 0);
genW0 = zeros(1, samples);
%% Generate Dictonary (random normal dist. generation)
genD = abs(randn(features, atoms));
%in each column, sets a random number of columns 0
for i=1:size(genD, 2)
    %generates the random integer
    r = randi(features - 1);
    %sets the first r rows to 0
    genD(1:r, i) = 0;
    %shuffles the whole row
    genD(:, i) = genD(randperm(length(genD(:, i))));
end
%% Generate Y   
genY = predictY(genD, genW, genW0);
end
