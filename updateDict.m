function [D, predictedY] = updateDict(Y, D, W, w0, lambda, initPredY, standardize)
    if nargin < 6 || isempty(initPredY)
        predictedY = predictY(D, W, w0); %predict Y for dictionary update
        %predictedY = center(predictedY);
    else
        predictedY = initPredY;
    end
    features = size(Y, 1);
    nrAtoms = size(D, 2);
    
    TOL = 1e-8;
    iter = 0;
    prevError = -inf;
    curError = calculateError(Y, D, W, w0, lambda);
    
    while (abs(curError - prevError) > TOL) && (iter < 100)
        %fprintf('Dict iteration: %i, Error: %.10f\n', iter, abs(curError - prevError));
        for f = 1:features
            for a = 1:nrAtoms
                %update predictedY first to take out current Dict atom
                predictedY(f, :) = predictedY(f, :) - D(f, a) * W(a, :);
                
                A = (Y(f, :) - predictedY(f, :)) * W(a, :)';
                B = W(a, :) * W(a, :)'+ 1e-8;

                %if (B == 0)
                %    B = B + 1e-8;
                %end
                
                %update Dictionary
                D(f, a) = A / B;
                %D(:, a) = center(D(:, a)) / max(norm(center(D(:, a))), 1);
                %predictedY(f, a) = D(f, a) * W(a, :)
                %predictedY(f, :) + D(f, a) * W(a, :)
                if standardize
                    D(:, a) = D(:, a) / max(norm(D(:, a), 2), 1);
                end
                %D(:, a) = normalize(D(:, a));
                
                %update predictedY for the new Dictionary atom
                predictedY(f, :) = predictedY(f, :) + D(f, a) * W(a, :);

                %if (B <= 0 || A < 0)
                %    D(f, a) = 0;
                %else
                %    D(f, a) = A / B;
                    %update predictedY for the new Dictionary Element
                %    predictedY(f, :) = predictedY(f, :) + D(f, a) * W(a, :);
                %end
            end
            %D(:, f) = normalize(D(:, f));
            %norm(D(:, f))
            %D(:, f) = center(D(:, f));
            %D(:, f) = center(D(:, f)) / max(norm(center(D(:, f))), 1);
        end
        %D = normalize(D);
        prevError = curError;
        curError = calculateError(Y, D, W, w0, lambda);
        iter = iter + 1;
    end
end