function [meanError, meanCost, meanSparsity] =... 
            CrossValidateDictLearn(k, init, lambda, genSizes,...
                                    randomSeed, hungarianTest)
    if isempty(init)
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
        [Y, ~, ~, ~] = genData({features, samples, nrAtoms}, 0, {60, 0}, {});
    else
       Y = init{1};
       features = size(Y, 1);
       samples = size(Y, 2);
       nrAtoms = features;
       if hungarianTest
            %assert(length(init) == 3, 'You must provide a Dictionary and weights.')
            %D = init{2};
            %W = init{3};
            costs = zeros(k, 1);
            wSparsities = zeros(k, 1);
       end
    end
    
    if isempty(lambda)
        lambda = 0.1;
    end
    if ~isempty(randomSeed)
       rng(randomSeed)
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
        [learnD, ~, ~] = DictionaryLearning(YTrain, lambda, nrAtoms, 500, false, false, {}, {});
        
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
            if hungarianTest
                learnedCorrelation = -abs(genD' * learnD);
                [~, learnCost] = munkres(learnedCorrelation);
                costs(i) = costs(i) + abs(learnCost);
                wSparsities(i) = wSparsities(i) + sum(sum(wTest(:, 2:end)==0)) / numel(wTest(:, 2:end))
            end
        end
    end
    parallelStop = toc(parallelCVStart);
    fprintf('%d - Fold test Finished. Time: %f\n', k ,parallelStop);
    
    meanError = sum(errors) / size(Y, 2);
    meanCost = sum(costs) / size(Y, 2);
    meanSparsity = sum(wSparsities) / size(Y, 2);
    %format long;
    %disp(meanError);
end