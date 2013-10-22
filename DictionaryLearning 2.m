function [ fitD, fitW, fitW0, lambda ] = DictionaryLearning( gY ,lambda, molNum, iterationNum )
%% check input sizes
% check gY
gYDim = size( gY );
assert( length( gYDim ) == 3 || length( gYDim ) == 2 );
assert( gYDim(1) > 1 );
% check lambda
if isempty( lambda )
    crossVali = true;
else
    crossVali = false;
    if( length( gYDim ) == 2 )
        assert( size( lambda, 1 ) == 1 );
    end
    if( length( gYDim ) == 3 )
        assert( size( lambda, 1 ) == gYDim(2) && size( lambda, 2 ) == gYDim(3) );
    end
end

%% initialization
% initialize sizes
sLen = gYDim(1);
if( length( gYDim ) == 3 )
    iHei = gYDim(2);
    iWid = gYDim(3);
end
if( length( gYDim ) == 2 )
    iHei = 1;
    iWid = 1;
end
mLen = molNum;
if isempty( lambda ) 
    lVal =  zeros( gYDim( 2 ), gYDim( 3 ) );
else
    lVal = lambda;
end
% fit variables initialize
fitD = abs( randn( sLen, mLen ) );
fitW = zeros( mLen, iHei, iWid );
fitW0 = zeros( iHei, iWid );

% iterated optimization
TOL = 1e-8;
ITLIMIT = iterationNum;
preTarget = -inf;
curTarget = computeObjf( gY, fitD, fitW, fitW0, lVal );
itNum = 1;

% generating geometric sequence
geoSeq = [0 0.0001 * ( 2.^[0:14] )];
cvAry = zeros( length( geoSeq ), 1 );

while ( abs( curTarget - preTarget ) > TOL ) && ( itNum <= ITLIMIT )
    %%update weights
    fprintf( '%d iterations\n', itNum );
    for i = 1:iHei
        for j = 1:iWid
            %% using matlab library
            %                 [B, FitInfo]= lassoglm( fitD, gY(:,i,j), 'normal', 'Lambda', lVal, 'RelTol', 1e-8 );
            %                 fitW(:,i,j) = B(:,1);
            %                 fitW0(i,j) = FitInfo.Intercept;
            %% cross validation with coordAsscentENet func
            if  crossVali
                k = 1;
                for m = geoSeq
                    cvAry(k) = generateCvObj( gY(:,i,j), fitD, m, {fitW0(i,j), fitW(:,i,j)}, 5 );
                    k = k + 1;
                end
                tmp = (cvAry == max( cvAry ) );
                if( length( tmp ) > 1 )
                    lVal(i, j) = geoSeq(1);
                else
                    lVal(i, j) = geoSeq(tmp);
                end
            end
            [beta0,beta] = coordAscentENet( gY(:,i,j), fitD, lVal(i,j), 0, {fitW0(i,j), fitW(:,i,j)} );
            fitW(:,i,j) = beta;
            fitW0(i,j) = beta0;
        end
    end
    preTarget = curTarget;
    curTarget = computeObjf( gY, fitD, fitW, fitW0, lVal );
    predictY = computePreY( fitD, fitW, fitW0 ); %%compute for dictionary update
    %fprintf( 'after lasso, curTarget: %e, preTarget: %e\n', curTarget, preTarget );
    if( abs( curTarget - preTarget ) > TOL )
        assert( curTarget - preTarget >= 0 );
    end
    %%update Dictionary
    for k = 1:sLen
        for r = 1:mLen
            a = fitW(r,:,:) .* ( gY(k,:,:) - predictY(k,:,:) + fitD(k,r) * fitW(r,:,:) );
            b = fitW(r,:,:).^2;
            a = squeeze( a );
            b = squeeze( b );
            a = sum( sum( a ) );
            b = sum( sum( b ) );
            if b == 0
                %update predictY first
                predictY(k,:,:) = predictY(k,:,:) - fitD( k, r ) * fitW( r, :, : );
                fitD(k,r) = 0;
            else
                %update predictY first
                predictY(k,:,:) = predictY(k,:,:) - fitD( k, r ) * fitW( r, :, : );
                if( a < 0 || b < 0 )
                    fitD(k,r) = 0;
                else
                    fitD(k,r) = a/b;
                    predictY(k,:,:) = predictY(k,:,:) + fitD( k, r ) * fitW( r, :, : );
                end
            end
        end
    end
    preTarget = curTarget;
    curTarget = computeObjf( gY, fitD, fitW, fitW0, lVal );
    itNum = itNum + 1;
    %fprintf( 'it num: %d curTarget: %e, preTarget: %e\n', itNum, curTarget, preTarget );
    if( abs( curTarget - preTarget ) > TOL )
        assert( curTarget - preTarget >= 0 );
    end
end

lambda = lVal;
end

