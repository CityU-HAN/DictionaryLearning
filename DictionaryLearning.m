function [Dict, W, w0] = DictionaryLearning(Y, lambda, nrAtoms, nrIterations, crossValidation, verbose, display, DInit)

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

%TODO: Implement corssValidation
if isempty(crossValidation)
    crossValidation = false;
end

if isempty(verbose)
    verbose = false;
end

if isempty(display)
    display = false;
end
%Initaliaze dictionary and weights
Dict = abs(randn(features, nrAtoms));
%Dict = DInit;
%W = zeros(nrAtoms, samples);
W = abs(randn(nrAtoms, samples));
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
        [w0Temp, WTemp] = coordAscentENet(Y(:,i), Dict, lambda, 0, {w0(i), W(:, i)}, 200);
        W(:, i) = WTemp;
        w0(i) = w0Temp;
    end
    
    %% Calculate and print imrpovements (error-difference)
    format long
    prevError = curError;
    curError = calculateError(Y, Dict, W, w0, lambda);
    errorDif = abs(curError - prevError); 
    
    if(verbose)
        fprintf('After lasso, curCost - prevCost = %.10f - %.10f = %.10f\n', curError, prevError, errorDif);
        fprintf('Updating Dictionary...\n');
    end
    
    if( abs(errorDif) > TOL )
        assert(errorDif >= 0);
    end
    
    %% Update Dict
    Dict = updateDict(Y, Dict, W, w0);
    
    %% Calculate and print imrpovements (error-difference)
    prevError = curError;
    curError = calculateError(Y, Dict, W, w0, lambda);
    errorDif = abs(curError - prevError);
    
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
    
    if(abs(errorDif) > TOL)
        assert(errorDif >= 0);
    end
    iter = iter + 1;
end
end