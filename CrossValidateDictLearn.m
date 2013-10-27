function [error]=CrossValidateDictLearn()
    features = 100;
    %Split half-half into two data sets
    samples = 160;
    nrAtoms = 50;
    %generates y R^features*samples
    [Y, D] = genData(features, samples, nrAtoms);
    yHalf = size(Y, 2)/2;
    
    yTrain = Y(:, 1:yHalf);
    yTest = Y(:, yHalf + 1:end);
    [assignment, cost] = munkres(yTrain);
    disp('Running Dictionary Learning Algorithm...');
    tic
    [fitD, fitW, fitW0] = DictionaryLearning(yTrain, 0.1, nrAtoms, 500, [], false, false);
    time = toc
    minutes = time/60
    predictedY = predictY(fitD, fitW, fitW0);
    disp('Mean, min, max of yTrain and predictedY:');
    disp([mean2(yTrain) mean2(predictedY)]);
    disp([min2(yTrain) min2(predictedY)]);
    disp([max2(yTrain) max2(predictedY)]);
    
    disp('Absolute and squared difference:');
    disp(sum(sum(abs(yTrain - predictedY))));
    disp(sum(sum((yTrain - predictedY).^2)));
    
    [assignmentNew, costNew] = munkres(predictedY);
    indicesEqual = assignment == assignmentNew
    
    assignment
    assignmentNew
    sum(1 - indicesEqual)
    cost
    costNew
    disp(abs(cost - costNew))
    return
    
    error = zeros(yHalf, 1);
    for i=1:yHalf
        wValidate = yTest(:, i) \ fitD;
        error(i) = sum((yTest(:, 1) - fitD * wValidate') .^ 2);
    end
    meanError = mean(error);
    format long;
    disp(meanError);
end