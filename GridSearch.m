    function [bestLambda]=GridSearch(k, Y, lambdaMax, genSizes, display)
    GridSearchStart = tic();
    if isempty(k)
        k = 5;
    end

    if isempty(genSizes)
        %features = 50;
        %samples = 200;
        %nrAtoms = 50;
        features = 20;
        samples = 40;
        nrAtoms = 20;
    else
        features = genSizes{1};
        samples = genSizes{2};
        nrAtoms = genSizes{3};
    end
    
    if isempty(Y)
        rng(1);
        %generates y R^features*samples
        [Y, D, ~, ~] = genData(features, samples, nrAtoms, {}, {}, {});
    end
    
    if isempty(lambdaMax)
        if isempty(D)
            lambdaMax = 2;
        else
            lambdaMax = max(max(Y,[],2) * max(max(D)));
        end
    end
    
    if isempty(display)
        display = false;
    end
    
    meanErrors = zeros(k, 1);
    lambdas = zeros(k, 1);
    
    fprintf('Starting Gridsearch for %i Lambdas, using %i-fold CrossValidation for each...\n', k*2, k);
    parfor i=1:k*2
        lambdas(i) = lambdaMax * 0.5^(i-1)* 0.5^(i-1);
        
        fprintf('\nIteration %i out of %i, testing Lambda=%f\n', i, (k*2), lambdas(i));
        
        meanErrors(i) = CrossValidateDictLearn(k, Y, lambdas(i), {});
        
        fprintf('MeanError: %i\n', meanErrors(i));
    end
    
    %get the index of the lambda with the minimum mean Error.
    bestLambda = lambdas(find(meanErrors == min(meanErrors)));
    
    if(display)
        lambdas
        meanErrors
        figure(1);
        fprintf('Plotting Errors of Grid Search of k-fold Cross validation')
        plot(meanErrors);
        xlabel('lambda #');
        ylabel('error');
        drawnow
        %drawnow ('x11', '/dev/null', false, 'gnuplotstream.gp') 
        save('dbg')
    else
        for i=1:k*2
            fprintf('Mean-error: %f for Lambda %f\n', meanErrors(i), lambdas(i));
        end
         fprintf('Best Lambda %f\n', bestLambda);
    end
    toc(GridSearchStart)
end