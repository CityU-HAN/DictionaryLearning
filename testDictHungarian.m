function [genCost, learnCost] = testDictHungarian(genSizes, lambda)
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
       lambda = 0.01;
    end
    
    [genY, genD] = genData(features, samples, nrAtoms, 0, -1, -1);
    genCorrelation = -abs(genD' * genD);
    trCorr = trace(genCorrelation)
    %[genAssign, genCost] = munkres(genCorrelation);
    
    fprintf('Dimensions of Y: %i features x %i samples.\n',...
        feautes, samples);
    fprintf('Lambda = %f, extracting %i atoms.\n', lambda, nrAtoms);
    disp('Learning the dictionary...');
    [learnD, learnW, learnW0] = ...
        DictionaryLearning(genY, lambda, nrAtoms, 200, [], true, false, genD);
    
    learnCorrelation = -abs(genD' * learnD);
    [learnAssign, learnCost] = munkres(learnCorrelation);
    expectedCost = abs(trCorr)
    learnedCost = abs(learnCost)
    difference - abs(genCost - learnCost)
    
    %genD
    %learnD
end