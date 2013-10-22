function [error]=CrossValidateDictLearn()
    features = 50;
    samples = 80;
    nrAtoms = 50;
    YTest = genData(features, samples, nrAtoms);
    YValidate = genData(features, samples, nrAtmos);
    
    [fitD, fitW, fitW0] = DictionaryLearningNew(generatedY, 0.1, 98, [], [], true, true);
    
    newW = YValidate \ fitD;
    
    error = sum((YValidate - (fitD * fitW + fitW0)) ^ 2)
end