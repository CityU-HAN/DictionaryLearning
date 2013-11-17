function [Dict, W, w0] = DictionaryLearning(Y, lambda, nrAtoms,...
                            nrIterations, verbose, display, DInit, WInit)

if nargin < 1
  error('Y is a required input')
end

% check gY has 2 dimensions
features = size(Y, 1);
samples = size(Y, 2);
assert(features > 1 && samples >= 1)

%check lambda
if isempty(lambda)
    lambda = 0.1;
end
%assert(lambda >= 0 && lambda <= 1)

if isempty(nrAtoms)
    nrAtoms = features;
end

if isempty(nrIterations)
    nrIterations = 1000;
end

if isempty(verbose)
    verbose = false;
end

if isempty(display)
    display = false;
end
%Initaliaze dictionary and weights
if isempty(DInit)
    Dict = randn(features, nrAtoms);
    W = randn(nrAtoms, samples);
    %[W, S, Dict] = svd(Y', 'econ');
    %Dict = bsxfun(@times, diag(S), Dict');
    %Dict = Dict';
    %W = W';
    %W = [ones(1, size(W, 2)); W];
    %W = randn(nrAtoms-1, samples);
    %Dict = repmat( mean(Y) / (nrAtoms*features), features, nrAtoms );
    %size(Dict)
else
    %Dict = DInit;
    %W = randn(nrAtoms, samples);
    [W, S, Dict] = svd(Y', 'econ');
    Dict = bsxfun(@times, diag(S), Dict');
    Dict = Dict';
    W = W';
end

w0 = zeros(1, samples);

%Loop-control variables
TOL = 1e-8;
MAXIT = nrIterations;
iter = 1;

%Setting the error, often also called cost
errors = zeros(MAXIT, 1);
prevError = -inf;
curError = calculateError(Y, Dict, W, w0, lambda);

format long
%format short

if(display)
    figure(1);
end

while (abs(curError - prevError) > TOL) && (iter <= MAXIT)
    if (verbose) 
        fprintf('Iteration: %i\n', iter);
        fprintf('Updating weights...\n');
    end
    %% Update each weight-vector
    for i = 1:samples
        %[WTemp, Fitinfo] = lassoglm(Dict, Y(:, i), 'normal', 'Lambda', lambda,...
        %    'RelTol', 1e-8, 'Standardize', false);
        %WTemp
        %Fitinfo.Intercept
        %W(:, i) = WTemp;
        %w0(i) = Fitinfo.Intercept;
        %return
        %lambda
        %WTemp = coordAscentENet(Y(:,i), [ones(features, 1) Dict], lambda, 0, {}, 200);
        [w0Temp, WTemp] = coordAscentENetIntercept(Y(:,i), Dict, lambda, 0, {w0(i), W(:, i)}, 200);
        %[WTest, ~] = larsen([ones(features, 1) Dict], Y(:,i), 0, lambda, [], false, false)
        %size([ones(features, 1) Dict])
        %size(Y(:, 1))
        %[WTest, ~] = larsen([ones(features, 1) zscore(Dict)], center(Y(:,i)), 0, lambda, [], false, false)
        %[WTest, ~] = larsen([ones(features, 1) Dict], Y(:,i), 0, lambda, [], false, false)
        %[WTest, ~] = larsen(Dict, Y(:,i), 0, lambda, [], false, false)
        %return
        %return
        %[wTest, ~] = larsen(Dict, Y(:,i), 0, lambda, [], false, false)
        %[wTest2, ~] = elasticnet([ones(size(Dict, 1), 1) Dict], Y(:,i), 0, lambda, false, false)
        %return
        %WTemp
        %return
        W(:, i) = WTemp;
        w0(i) = w0Temp;
        %W(:, i) = WTemp(2:end);
        %w0(i) = WTemp(1);
    end
    
    %% Calculate and print imrpovements (error-difference)
    format long
    %prevError = curError;
    %curError = calculateError(Y, Dict, W, w0, lambda);
    %errorDif = curError - prevError; 
    
    if(verbose)
        %fprintf('After lasso, curCost - prevCost = %.10f - %.10f = %.10f\n', curError, prevError, errorDif);
        fprintf('Updating Dictionary...\n');
    end
    
    %if( abs(errorDif) > TOL )
    %    assert(errorDif >= 0);
    %end
    
    %% Update Dict
    Dict = updateDict(Y, Dict, W, w0, lambda);
    
    %% Calculate and print imrpovements (error-difference)
    prevError = curError;
    curError = calculateError(Y, Dict, W, w0, lambda);
    errorDif = curError - prevError;
    
    if(verbose)
        fprintf('Dictionary updated, curCost - prevCost = %.10f - %.10f = %.10f\n', curError, prevError, errorDif);
    end
    
    %% Plot calculated errors for each iteration
    if(display)
        errors(iter) = curError;
        if mod(iter, 10) == 0
            fprintf('Plotting Errors')
            plot(errors(1:iter));
            xlabel('iteration');
            ylabel('error');
            drawnow
            %drawnow ('x11', '/dev/null', false, 'gnuplotstream.gp') 
            save('dbg')
        end
    end
    
    if(abs(errorDif) > TOL && errorDif < 0)
        %assert(errorDif >= 0);
         fprintf('Whooop!, curCost - prevCost = %.10f - %.10f = %.10f\n', curError, prevError, errorDif);
    end
    iter = iter + 1;
end
end