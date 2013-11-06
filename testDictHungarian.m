function [genCost, learnCost] = testDictHungarian(genSizes)
    if isempty(genSizes)
        features = 8;
        samples = 20;
        nrAtoms = 8;
    else
        features = genSizes{1};
        samples = genSizes{1};
        nrAtoms = genSizes{1};
    end
    [genY, genD] = genData(features, samples, nrAtoms, 0, -1, -1);
    genD
    genCorrelation = -abs(genD' * genD);
    trCorr = trace(genCorrelation)
    [genAssign, genCost] = munkres(genCorrelation);
    
    disp('learning dictionary');
    [learnD, learnW, learnW0] = ...
        DictionaryLearning(genY, 0.1, nrAtoms, 200, [], true, false, genD);
    
    learnCorrelation = -abs(genD' * learnD);
    [learnAssign, learnCost] = munkres(learnCorrelation);
    expectedCost = abs(genCost)
    learnedCost = abs(learnCost)
    difference - abs(genCost - learnCost)
    
    %genD
    %learnD
end