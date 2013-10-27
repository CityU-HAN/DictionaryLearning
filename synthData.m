clc
%% number of features/measurements of a patient
features = 50;
%% number of samples/patients
samples = 98;
% number of dictionary-atoms, <= samples (the bigger, the closer the approximation is
% to Y)
atoms = 50;
%% Generate w: All values bigger than 0 will be 1, all smaller 0
generatedW = (randn(atoms, samples) > 0);
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
%y with noise
%gY = predictY(generatedD, generatedW, generatedW0) + 0.1 *
%abs(randn(features, samples))
%% fitting without noise
fprintf('Learned paramaters from noise-free synthetic data\n');
size(generatedY)
%tic
%[ fitD, fitW, fitW0 ] = DictionaryLearningNew(generatedY, 0.1, 98, [], [], true, true)
%size(fitD)
%size(fitW)
%toc
