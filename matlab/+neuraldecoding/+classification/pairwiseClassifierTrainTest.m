function result = pairwiseClassifierTrainTest( ...
    X, trialTypes, trainMask, testMask, negativeClass, positiveClass, thirdClass)
%PAIRWISECLASSIFIERTRAINTEST Fit on one run and score a separate run.

trialTypes = lower(string(trialTypes(:)));
trainMask = logical(trainMask(:));
testMask = logical(testMask(:));
negativeClass = lower(string(negativeClass));
positiveClass = lower(string(positiveClass));
thirdClass = lower(string(thirdClass));

trainPairMask = trainMask & ...
    (trialTypes == negativeClass | trialTypes == positiveClass);
testPairMask = testMask & ...
    (trialTypes == negativeClass | trialTypes == positiveClass);
thirdMask = testMask & trialTypes == thirdClass;
yTrain = double(trialTypes(trainPairMask) == positiveClass);
yTest = double(trialTypes(testPairMask) == positiveClass);

result = struct;
result.valid = false;
result.message = "";
result.negativeClass = negativeClass;
result.positiveClass = positiveClass;
result.thirdClass = thirdClass;
result.trainingClassCounts = [sum(yTrain == 0), sum(yTrain == 1)];
result.testClassCounts = [sum(yTest == 0), sum(yTest == 1)];
result.score = NaN(size(X, 1), 1);
result.prediction = NaN(size(X, 1), 1);
result.accuracy = NaN;
result.balancedAccuracy = NaN;
result.auc = NaN;
result.confusionMatrix = NaN(2, 2);
result.thirdClassFractionPositive = NaN;
result.thirdClassMedianScore = NaN;
result.model = [];
result.preprocessing = struct;

if any(result.trainingClassCounts < 2)
    result.message = "Training data require at least two trials per class.";
    return;
end
if any(result.testClassCounts < 1)
    result.message = "Test data must contain both evaluation classes.";
    return;
end

center = mean(X(trainPairMask, :), 1);
scale = std(X(trainPairMask, :), 0, 1);
scale(scale == 0 | ~isfinite(scale)) = 1;
XTrain = (X(trainPairMask, :) - center) ./ scale;
model = fitclinear(XTrain, yTrain, ...
    'Learner', 'logistic', ...
    'Regularization', 'ridge', ...
    'ClassNames', [0 1]);

scoreMask = testPairMask | thirdMask;
XZ = (X(scoreMask, :) - center) ./ scale;
result.score(scoreMask) = XZ * model.Beta + model.Bias;
result.prediction(scoreMask) = result.score(scoreMask) >= 0;

pairScores = result.score(testPairMask);
pairPredictions = double(pairScores >= 0);
result.accuracy = mean(pairPredictions == yTest);
result.confusionMatrix = confusionmat(yTest, pairPredictions, 'Order', [0 1]);
support = sum(result.confusionMatrix, 2);
recall = diag(result.confusionMatrix) ./ max(support, 1);
recall(support == 0) = NaN;
result.balancedAccuracy = mean(recall, 'omitnan');
[~, ~, ~, result.auc] = perfcurve(yTest, pairScores, 1);
if any(thirdMask)
    result.thirdClassFractionPositive = mean(result.score(thirdMask) >= 0);
    result.thirdClassMedianScore = median(result.score(thirdMask), 'omitnan');
end
result.model = model;
result.preprocessing = struct('center', center, 'scale', scale);
result.valid = true;
end
