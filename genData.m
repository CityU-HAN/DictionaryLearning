function generatedY = genData(features, samples, atoms)
%% Generate w: All values bigger than 0 will be 1, all smaller 0
generatedW = randn(atoms, samples);
generatedW = generatedW .* (generatedW > 0);
generatedW0 = zeros(1, samples);
%% Generate Dictonary (random normal dist. generation)
generatedD = abs(randn(features, atoms));
%in each column, sets a random number of columns 0
for i=1:size(generatedD, 2)
    %generates the random integer
    r = randi(features - 1);
    %sets the first r rows to 0
    generatedD(1:r, i) = 0;
    %shuffles the whole row
    generatedD(:, i) = generatedD(randperm(length(generatedD(:, i))));
end
%% Generate Y   
generatedY = predictY(generatedD, generatedW, generatedW0);
end
