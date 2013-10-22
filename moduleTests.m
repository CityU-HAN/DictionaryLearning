classdef moduleTests < matlab.unittest.TestCase 

    methods(Test)
        function testLasso(testCase)
            beta0Expected = 1; betaExpected = [1 1 1 1 0 0 0 0]';
            N=100;
            p = length(betaExpected);
            x = rand(N, p);
            y = beta0Expected + x*betaExpected;
            [beta0, beta] = coordAscentENet(y, x, 0.1, 0, {}, {});
            testCase.assertSize(beta, [p 1]);
            testCase.assertEqual(beta0, beta0Expected, 'AbsTol', 0.05);
            testCase.assertEqual(beta, betaExpected, 'AbsTol', 0.05);
            %For a complete list see:
            %http://www.mathworks.com/help/matlab/matlab_prog/select-qualifications.html
        end

        function testDictUpdate(testCase)
            %Will test the DictUpdate soon
        end
        
        function testDictionaryLearning(testCase)
            %Will test the DictionaryLearning soon
        end
    end
    
 end
