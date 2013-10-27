function D = updateDict2(Y, D, W, w0)
    predictedY = predictY(D, W, w0); %predict Y for dictionary update
    features = size(Y, 1);
    nrAtoms = size(D, 2);
    for a = 1:nrAtoms
        A = (Y - predictedY + D(:, a) * W(a, :)) * W(a, :)';
        B = W(a, :) * W(a, :)';
        %update predictedY first
        predictedY = predictedY - D(:, a) * W(a, :);
        if (B == 0)
            D(:, a) = 0;
        else
            D(:, a) = A / B;
            %update predictedY for the new Dictionary Element
            predictedY = predictedY + D(:, a) * W(a, :);
        end
    end
end