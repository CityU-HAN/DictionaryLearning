function D = updateDict(Y, D, W, w0, lambda)
    predictedY = predictY(D, W, w0); %predict Y for dictionary update
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
                A = (Y(f, :) - predictedY(f, :) + D(f, a) * W(a, :)) * W(a, :)';
                B = W(a, :) * W(a, :)'+ 1e-8;
                %update predictedY first
                predictedY(f, :) = predictedY(f, :) - D(f, a) * W(a, :);

                %if (B == 0)
                %    B = B + 1e-8;
                %end

                D(f, a) = A / B;
                %update predictedY for the new Dictionary Element
                predictedY(f, :) = predictedY(f, :) + D(f, a) * W(a, :);

                %if (B <= 0 || A < 0)
                %    D(f, a) = 0;
                %else
                %    D(f, a) = A / B;
                    %update predictedY for the new Dictionary Element
                %    predictedY(f, :) = predictedY(f, :) + D(f, a) * W(a, :);
                %end
            end
            %D(:, f) = D(:, f) / max(norm(D(:, f), 2), 1);
        end
        prevError = curError;
        curError = calculateError(Y, D, W, w0, lambda);
        iter = iter + 1;
    end
end