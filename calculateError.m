function [cost] = calculateError(Y, D, W, w0, lambda)
%We are optimizing/maximizing the cost function -0.5 * ||Y - D*W||_2^2 + lambda*||W||_1
%lambda is a scalar right now. Could also be a vector in the future
    residuals = Y - predictY(D, W, w0);
    cost = -0.5 * sum(sum(residuals .^ 2)) - lambda * sum(sum(abs(W)));
end

