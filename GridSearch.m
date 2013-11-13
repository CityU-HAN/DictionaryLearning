function [bestLambda]=GridSearch(k, init, lambdaMax, genSizes, randomSeed, hungarianTest, display)
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
    
    if isempty(init)
        rng(1);
        %generates y R^features*samples
        [Y, D, W, ~] = genData({features, samples, nrAtoms}, 0, {60, 0}, {});
    else
        Y = init{1};
        D = init{2};
        if length(init) > 2
            W = init{3};
        end
        if length(init) > 3
            W0 = init{4};
        end
        
        features = size(Y, 1);
        samples = size(Y, 2);
        nrAtoms = size(D, 2);
    end
    
    if isempty(lambdaMax)
        if isempty(D)
            lambdaMax = 2;
        else
            lambdaMax = 0;
            for j=1:samples
                localMax = max(abs(Y(:, j)' * D));
                lambdaMax = max(localMax, lambdaMax);
            end
            %lambdaMax = max(max(Y,[],2) * max(max(D)));
        end
    end
    
    if isempty(display)
        display = false;
    end
    
    meanErrors = zeros(k, 1);
    meanCosts = zeros(k, 1);
    meanSparsities = zeros(k, 1);
    lambdas = zeros(k, 1);
    
    fprintf('Starting Gridsearch for %i Lambdas, using %i-fold CrossValidation for each...\n', k*2, k);
    parfor i=1:k*2
        lambdas(i) = lambdaMax * 0.1^(i-1);
        
        fprintf('\nIteration %i out of %i, testing Lambda=%f\n', i, (k*2), lambdas(i));
        
        [meanErrors(i), meanCosts(i), meanSparsities(i)] = ...
            CrossValidateDictLearn(k, {Y, D}, lambdas(i), {}, randomSeed, hungarianTest);
        
        fprintf('MeanError: %i\n', meanErrors(i));
    end
    
    %get the index of the lambda with the minimum mean Error.
    bestLambda = lambdas(find(meanErrors == min(meanErrors))); 
    
    
    if(display)
        lambdas
        meanErrors
        figure(1);
        fprintf('Plotting Errors of Grid Search of k-fold Cross validation');
        plot(meanErrors);
        xlabel('lambda #');
        ylabel('error');
        drawnow
        %drawnow ('x11', '/dev/null', false, 'gnuplotstream.gp') 
        save('dbg')
    else
        disp('----------------\nGround Truth: %f\n');
        expectedCost = abs(trace(-abs(genD' * genD)));
        fprintf('Expected Cost: %f\n', expectedCost)
        fprintf('Sparsity of W: %f\n', sum(sum(W==0)) / numel(W))
        for i=1:k*2
            fprintf('Lambda: %f\n', lambdas(i));
            fprintf('Mean-error: %f\n', meanErrors(i));
            fprintf('Mean-Cost (Hungarian): %f, Difference: %f\n', meanCosts(i), abs(expectedCost - meanCosts(i)));
            fprintf('Mean Sparsity of W: %f\n\n', meanSparsities(i));
        end
         fprintf('Best Lambda %f\n', bestLambda);
    end
    toc(GridSearchStart)
end