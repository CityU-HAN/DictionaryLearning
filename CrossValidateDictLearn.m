function [meanError]=CrossValidateDictLearn(k, Y, lambda, genSizes)
    if isempty(Y)
        if isempty(genSizes)
            features = 50;
            samples = 100;
            nrAtoms = 50;
        else
            features = genSizes{1};
            samples = genSizes{2};
            nrAtoms = genSizes{3};
        end
        rng(1);
        %generates y R^features*samples
        [Y, ~, ~, ~] = genData(features, samples, nrAtoms, 0, 0.5, {});
    else
       features = size(Y, 1);
       samples = size(Y, 2);
       nrAtoms = features;
    end
    
    if isempty(lambda)
        lambda = 0.1;
    end

    indices = crossvalind('Kfold', samples, k);
    fprintf('\nRunning Dictionary Learning Cross Validation.\n');
    fprintf('Chosen Lambda = %f, number of Dict atoms: %i\n\n', lambda, nrAtoms);
    
    errors = zeros(k, 1);
    parallelCVStart = tic();
    parfor i=1:k
        YTrain = Y(:, indices ~= i);
        YTest = Y(:, indices == i);
        
        testSamples = size(YTest, 2);
        
        fprintf('Size training-set: %i features x %i samples. Test-set: %i features x %i samples.\n',...
            size(YTrain, 1), size(YTrain, 2), size(YTest, 1), testSamples);
        
        fprintf('Training on set %i out of %i...\n', i, k);
        %tic;
        [learnD, ~, ~] = DictionaryLearning(YTrain, lambda, nrAtoms, 500, [], false, false, {});
        %time = toc;
        %minutes = time/60;
        %fprintf('Running time %.10f minutes\n', minutes);
        
        %Adding a column of 1s to the learned Dictionary
        learnD = [ones(size(learnD, 1), 1) learnD];
        
        %error = zeros(testSamples, k);
        fprintf('Testing on set %i out of %i...\n', i, k);
        for j=1:testSamples
            wTest = YTest(:, j) \ learnD;
            errors(i) = errors(i) + sum((YTest(:, j) - learnD * wTest') .^ 2);
        end
    end
    parallelStop = toc(parallelCVStart);
    fprintf('Time: %f\', parallelStop);
    
    meanError = sum(errors) / size(Y, 2);
    %format long;
    %disp(meanError);
end