function result = pairwiseClassifierCV( ...
    X, trialTypes, negativeClass, positiveClass, thirdClass, k, randomSeed)
%PAIRWISECLASSIFIERCV Out-of-fold binary scores plus unseen third-class scores.

if nargin < 6 || isempty(k)
    k = 5;
end
if nargin < 7 || isempty(randomSeed)
    randomSeed = 1;
end

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
    error('pairwiseClassifierCV:SizeMismatch', ...
        'Rows of X must match the number of trial labels.');
end
if effectiveFolds < 2
    result.message = "At least two trials from each training class are required.";
    return;
end

rng(randomSeed, 'twister');
pairIndices = find(pairMask);
thirdIndices = find(thirdMask);
partition = cvpartition(categorical(y), ...
    'KFold', effectiveFolds, 'Stratify', true);
thirdFoldScores = NaN(numel(thirdIndices), effectiveFolds);

for fold = 1:effectiveFolds
    localTrain = training(partition, fold);
    localTest = test(partition, fold);
    trainIndices = pairIndices(localTrain);
    testIndices = pairIndices(localTest);

    fitted = fitPairwiseModel(X(trainIndices, :), y(localTrain));
    result.score(testIndices) = applyPairwiseModel(fitted, X(testIndices, :));
    result.foldIndex(testIndices) = fold;
    result.models{fold} = fitted.model;
    result.preprocessing{fold} = rmfield(fitted, 'model');

    if ~isempty(thirdIndices)
        thirdFoldScores(:, fold) = applyPairwiseModel( ...
            fitted, X(thirdIndices, :));
    end
end

if ~isempty(thirdIndices)
    result.score(thirdIndices) = mean(thirdFoldScores, 2, 'omitnan');
    result.thirdClassFoldScores = thirdFoldScores;
end

result.prediction(pairMask | thirdMask) = ...
    result.score(pairMask | thirdMask) >= 0;
result = calculateMetrics(result, y, pairIndices, thirdIndices);
result.valid = true;
result.message = "";
end


function result = emptyResult(numberTrials, negativeClass, positiveClass, ...
    thirdClass, classCounts, folds)
result = struct;
result.valid = false;
result.message = "";
result.negativeClass = negativeClass;
result.positiveClass = positiveClass;
result.thirdClass = thirdClass;
result.classCounts = classCounts;
result.numberFolds = folds;
result.score = NaN(numberTrials, 1);
result.prediction = NaN(numberTrials, 1);
result.foldIndex = NaN(numberTrials, 1);
result.models = cell(folds, 1);
result.preprocessing = cell(folds, 1);
result.thirdClassFoldScores = [];
result.accuracy = NaN;
result.balancedAccuracy = NaN;
result.auc = NaN;
result.confusionMatrix = NaN(2, 2);
result.thirdClassFractionPositive = NaN;
result.thirdClassMedianScore = NaN;
end


function fitted = fitPairwiseModel(X, y)
center = mean(X, 1);
scale = std(X, 0, 1);
scale(scale == 0 | ~isfinite(scale)) = 1;
XZ = (X - center) ./ scale;
model = fitclinear(XZ, y, ...
    'Learner', 'logistic', ...
    'Regularization', 'ridge', ...
    'ClassNames', [0 1]);
fitted = struct('model', model, 'center', center, 'scale', scale);
end


function score = applyPairwiseModel(fitted, X)
XZ = (X - fitted.center) ./ fitted.scale;
score = XZ * fitted.model.Beta + fitted.model.Bias;
end


function result = calculateMetrics(result, y, pairIndices, thirdIndices)
pairScores = result.score(pairIndices);
pairPredictions = double(pairScores >= 0);
result.accuracy = mean(pairPredictions == y);
result.confusionMatrix = confusionmat(y, pairPredictions, 'Order', [0 1]);
support = sum(result.confusionMatrix, 2);
recall = diag(result.confusionMatrix) ./ max(support, 1);
recall(support == 0) = NaN;
result.balancedAccuracy = mean(recall, 'omitnan');
[~, ~, ~, result.auc] = perfcurve(y, pairScores, 1);
if ~isempty(thirdIndices)
    result.thirdClassFractionPositive = mean(result.score(thirdIndices) >= 0);
    result.thirdClassMedianScore = median(result.score(thirdIndices), 'omitnan');
end
end
