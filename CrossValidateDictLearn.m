function [error]=CrossValidateDictLearn()
    features = 50;
    samples = 80;
    nrAtoms = 50;
    %generates y R^features*samples
    Y = genData(features, samples, nrAtoms);
    yHalf = size(Y, 2)/2;
    
    yTest = Y(:, 1:yHalf);
    yValidate = Y(:, yHalf + 1:end);
    
    [fitD, fitW, fitW0] = DictionaryLearning(yTest, 0.1, 50, 1000, [], true, false);
    
    error = zeros(yHalf, 1);
    for i=1:yHalf
        wValidate = yValidate(:, i) \ fitD;
        error(i) = sum((yValidate(:, 1) - fitD * wValidate') .^ 2);
    end
    sum(error)
end