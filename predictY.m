function predictedY = predictY(D, W, w0 )
    %predicts/approximates Y by linearly combining the weights with the
    %Dictionary/basis vectors: D*w
    
    %preY = zeros( size( D, 1 ), size( W, 2 ) );
    %preY = w0 + D * W
    %bsxfun is just necessary for adding w0
    predictedY = bsxfun(@plus, D * W, w0);
end
