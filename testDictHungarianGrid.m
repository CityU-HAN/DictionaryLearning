function [expectedCost, learnedCost] = testDictHungarianGrid(k, Y,...
                        lambdaMax, genSizes, randomSeed, display)
    if isempty(genSizes)
        features = 8;
        samples = 20;
        nrAtoms = 8;
    else
        features = genSizes{1};
        samples = genSizes{1};
        nrAtoms = genSizes{1};
    end
    
    if isempty(lambda)
       lambda = 0;
    end
    
    if ~isempty(randomSeed)
        rng(randomSeed)
    end
    
    if isempty(init)
        if ~isempty(randomSeed)
            rng(randomSeed)
        end
        [genY, genD, genW, ~] = genData(features, samples, nrAtoms, 0, 0.6, 0, 10);
    else
        genY = init{1};
        genD = init{2};
        if length(init) > 2
            genW = init{3};
        end
        if length(init) > 3
            genW0 = init{4};
        end
    end
    genCorrelation = -abs(genD' * genD);
    trCorr = trace(genCorrelation);
    
    fprintf('Dimensions of Y: %i features x %i samples.\n',...
        features, samples);
    fprintf('Lambda = %f, extracting %i atoms.\n', lambda, nrAtoms);
    disp('Learning the dictionary...');
    GridSearch(k, Y, lambdaMax, genSizes, randomSeed, display);
    
    learnedCorrelation = -abs(genD' * learnD);
    [learnedAssign, learnedCost] = munkres(learnedCorrelation);
    expectedCost = abs(trCorr)
    learnedCost = abs(learnedCost)
    difference = abs(expectedCost - learnedCost)
end