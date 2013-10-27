classdef moduleTests < matlab.unittest.TestCase 

    methods(Test)
        function testLasso(testCase)
            beta0Expected = 1; betaExpected = [1 1 1 1 0 0 0 0]';
            N=100;
            p = length(betaExpected);
            x = rand(N, p);
            y = beta0Expected + x*betaExpected;
            [beta0, beta] = coordAscentENet(y, x, 0.1, 0, {}, {});
            testCase.fatalassertSize(beta, [p 1]);
            testCase.fatalassertEqual(beta0, beta0Expected, 'AbsTol', 0.05);
            testCase.fatalassertEqual(beta, betaExpected, 'AbsTol', 0.05);
            %For a complete list see:
            %http://www.mathworks.com/help/matlab/matlab_prog/select-qualifications.html
        end

        function testDictUpdate(testCase)
            %features = 100;
            %samples = 80;
            %nrAtoms = 50;
            %[Y, DictExpected] = genData(features, samples, nrAtoms);
            %Initaliaze dictionary and weights
            %Dict = abs(randn(features, nrAtoms));
            %W = abs(randn(nrAtoms, samples));
            %w0 = zeros(1, samples);
            %lambda = 0;
            %TOL = 1e-8;
            %MAXIT = 500;
            %iter = 1;
            
            %prevError = -inf;
            %curError = calculateError(Y, Dict, W, w0, lambda);
            %while (abs(curError - prevError) > TOL) && (iter <= MAXIT)
            %    Dict = updateDict(Y, Dict, W, w0);
            %    prevError = curError;
            %    curError = calculateError(Y, Dict, W, w0, lambda);
            %    iter = iter + 1;
            %end
            
            %testCase.assertSize(Dict, [features nrAtoms]);
            
            %[assignmentExpected, costExpected] = munkres(DictExpected);
            %costExpected
            %assignmentExpected
            %[assignment, cost] = munkres(Dict);
            %cost
            %assignment
            %testCase.assertEqual(cost, costExpected);
            %testCase.assertEqual(assignment, assignmentExpected);
        end
        
        function testDictionaryLearning(testCase)
            features = 100;
            samples = 80;
            nrAtoms = 50;
            [Y, ~] = genData(features, samples, nrAtoms);
            [D, W, W0] = DictionaryLearning(yTrain, 0, nrAtoms, 500, [], false, false);
            
            testCase.assertSize(Dict, [features nrAtoms]);
            
            [assignmentExpected, costExpected] = munkres(yTrain);
            [assignment, cost] = munkres(predictY(fitD, fitW, fitW0));
            indicesEqual = assignmentExpected == assignment;
            
            testCase.assertEqual(cost, costExpected, 'AbsTol', 0.6);
            testCase.assertEqual(0, sum(1 - indicesEqual), 'AbsTol', 4);

        end
    end
    
 end
