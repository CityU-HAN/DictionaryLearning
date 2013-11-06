function testDictHungarian = testDictHungarian()
    features = 8;
    samples = 20;
    nrAtoms = 8;
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
    abs(genCost)
    abs(learnCost)
    
    %genD
    %learnD
    
    