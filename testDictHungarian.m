function [expectedCost, learnedCost] = testDictHungarian(init, genSizes, lambda, randomSeed, verbose)
    if isempty(genSizes)
        features = 8;
        samples = 20;
        nrAtoms = 9;
    else
        features = genSizes{1};
        samples = genSizes{2};
        nrAtoms = genSizes{3};
    end
    
    if isempty(lambda)
       lambda = 0.1;
    end
    
    if ~isempty(randomSeed)
        rng(randomSeed)
    end
    
    if isempty(verbose)
       verbose = false; 
    end
    
    if isempty(init)
        [genY, genD, genW, ~] = genData({features, samples, nrAtoms}, 0, {60, 0, 10});
    else
        genY = init{1};
        genD = init{2};
        if length(init) > 2
            genW = init{3};
        end
        if length(init) > 3
            genW0 = init{4};
        end
        
        features = size(genY, 1);
        samples = size(genY, 2);
        nrAtoms = size(genD, 2);
    end
    genCorrelation = -abs(genD' * genD);
    trCorr = trace(genCorrelation)
    %[genAssign, genCost] = munkres(genCorrelation);
    
    fprintf('Dimensions of Y: %i features x %i samples.\n',...
        features, samples);
    fprintf('Lambda = %f, extracting %i atoms.\n', lambda, nrAtoms);
    disp('Learning the dictionary...');
    [learnD, learnW, learnW0] = ...
        DictionaryLearning(genY, lambda, nrAtoms, 200, verbose, false, {}, {});
    
    genWSparsityLevel = sum(sum(genW<0.01 & genW>-0.01)) / numel(genW)
    learnWSparsityLevel = sum(sum(learnW<0.01 & learnW>-0.01)) / numel(learnW)
    
    colGenWSparsity = sum(genW<0.01 & genW>-0.01) / nrAtoms
    colLearnWSparsity = sum(learnW<0.01 & learnW>-0.01) / nrAtoms
    
    learnedCorrelation = -abs(genD' * learnD);
    [learnedAssign, learnedCost] = munkres(learnedCorrelation);
    expectedCost = abs(trCorr)
    learnedCost = abs(learnedCost)
    difference = abs(expectedCost - learnedCost)
end