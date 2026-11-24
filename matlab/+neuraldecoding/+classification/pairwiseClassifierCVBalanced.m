function result = pairwiseClassifierCVBalanced( ...
    X, trialTypes, negativeClass, positiveClass, thirdClass, k, randomSeed)
%PAIRWISECLASSIFIERCVBALANCED Logistic CV with equal total class weight.
% Scores are logits. waterProbability is the fold-specific logistic
% probability for the positive class; unseen-class probabilities are
% averaged across the fold-specific models.

if nargin < 6 || isempty(k), k = 5; end
if nargin < 7 || isempty(randomSeed), randomSeed = 1; end

trialTypes = lower(string(trialTypes(:)));
negativeClass = lower(string(negativeClass));
positiveClass = lower(string(positiveClass));
thirdClass = lower(string(thirdClass));
pairMask = trialTypes == negativeClass | trialTypes == positiveClass;
thirdMask = trialTypes == thirdClass;
y = double(trialTypes(pairMask) == positiveClass);
classCounts = [sum(y == 0), sum(y == 1)];
effectiveFolds = min([k, classCounts]);

result = emptyResult(size(X, 1), negativeClass, positiveClass, ...
    thirdClass, classCounts, effectiveFolds);
if size(X, 1) ~= numel(trialTypes)
    error('pairwiseClassifierCVBalanced:SizeMismatch', ...
        'Rows of X must match the number of trial labels.');
end
if effectiveFolds < 2
    result.message = "At least two trials from each class are required.";
    return;
end

rng(randomSeed, 'twister');
pairIndices = find(pairMask);
thirdIndices = find(thirdMask);
partition = cvpartition(categorical(y), 'KFold', effectiveFolds, ...
    'Stratify', true);
thirdFoldScores = NaN(numel(thirdIndices), effectiveFolds);
thirdFoldProbabilities = NaN(numel(thirdIndices), effectiveFolds);

for fold = 1:effectiveFolds
    localTrain = training(partition, fold);
    localTest = test(partition, fold);
    trainIndices = pairIndices(localTrain);
    testIndices = pairIndices(localTest);
    fitted = fitBalancedModel(X(trainIndices, :), y(localTrain));
    score = applyModel(fitted, X(testIndices, :));
    result.score(testIndices) = score;
    result.waterProbability(testIndices) = logistic(score);
    result.foldIndex(testIndices) = fold;
    if ~isempty(thirdIndices)
        score = applyModel(fitted, X(thirdIndices, :));
        thirdFoldScores(:, fold) = score;
        thirdFoldProbabilities(:, fold) = logistic(score);
    end
end

if ~isempty(thirdIndices)
    result.score(thirdIndices) = mean(thirdFoldScores, 2, 'omitnan');
    result.waterProbability(thirdIndices) = ...
        mean(thirdFoldProbabilities, 2, 'omitnan');
    result.thirdClassFoldScores = thirdFoldScores;
    result.thirdClassFoldProbabilities = thirdFoldProbabilities;
end
result.prediction(pairMask | thirdMask) = ...
    result.waterProbability(pairMask | thirdMask) >= 0.5;
pairScores = result.score(pairIndices);
pairPredictions = double(result.prediction(pairIndices));
result.confusionMatrix = confusionmat(y, pairPredictions, 'Order', [0 1]);
support = sum(result.confusionMatrix, 2);
recall = diag(result.confusionMatrix) ./ max(support, 1);
recall(support == 0) = NaN;
result.balancedAccuracy = mean(recall, 'omitnan');
[~, ~, ~, result.auc] = perfcurve(y, pairScores, 1);
if ~isempty(thirdIndices)
    result.thirdClassMeanWaterProbability = ...
        mean(result.waterProbability(thirdIndices), 'omitnan');
    result.thirdClassMedianWaterProbability = ...
        median(result.waterProbability(thirdIndices), 'omitnan');
end
result.valid = true;
result.message = "";
end

function fitted = fitBalancedModel(X, y)
center = mean(X, 1);
scale = std(X, 0, 1);
scale(scale == 0 | ~isfinite(scale)) = 1;
XZ = (X - center) ./ scale;
counts = [sum(y == 0), sum(y == 1)];
weights = zeros(size(y));
weights(y == 0) = 1 / counts(1);
weights(y == 1) = 1 / counts(2);
weights = weights * numel(y) / sum(weights);
model = fitclinear(XZ, y, 'Learner', 'logistic', ...
    'Regularization', 'ridge', 'ClassNames', [0 1], 'Weights', weights);
fitted = struct('model', model, 'center', center, 'scale', scale);
end

function score = applyModel(fitted, X)
score = ((X - fitted.center) ./ fitted.scale) * ...
    fitted.model.Beta + fitted.model.Bias;
end

function probability = logistic(score)
probability = zeros(size(score));
positive = score >= 0;
probability(positive) = 1 ./ (1 + exp(-score(positive)));
exponent = exp(score(~positive));
probability(~positive) = exponent ./ (1 + exponent);
end

function result = emptyResult(numberTrials, negativeClass, positiveClass, ...
    thirdClass, classCounts, folds)
result = struct('valid', false, 'message', "", ...
    'negativeClass', negativeClass, 'positiveClass', positiveClass, ...
    'thirdClass', thirdClass, 'classCounts', classCounts, ...
    'numberFolds', folds, 'score', NaN(numberTrials, 1), ...
    'waterProbability', NaN(numberTrials, 1), ...
    'prediction', NaN(numberTrials, 1), ...
    'foldIndex', NaN(numberTrials, 1), ...
    'thirdClassFoldScores', [], 'thirdClassFoldProbabilities', [], ...
    'balancedAccuracy', NaN, 'auc', NaN, ...
    'confusionMatrix', NaN(2, 2), ...
    'thirdClassMeanWaterProbability', NaN, ...
    'thirdClassMedianWaterProbability', NaN);
end
