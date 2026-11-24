function result = trialIndexRegressionCV( ...
    X, trialProgress, k, randomSeed, maxComponents)
%TRIALINDEXREGRESSIONCV Predict normalized trial progress with CV.
%
% This extends linearRegressionCV by retaining every held-out prediction,
% fold assignment, fitted model, and preprocessing transform.

if nargin < 3 || isempty(k)
    k = 5;
end
if nargin < 4 || isempty(randomSeed)
    randomSeed = 1;
end
if nargin < 5 || isempty(maxComponents)
    maxComponents = 10;
end

trialProgress = trialProgress(:);
numberObservations = size(X, 1);
numberAvailableComponents = size(X, 2);
if numberObservations ~= numel(trialProgress)
    error('trialIndexRegressionCV:SizeMismatch', ...
        'Rows of X must match the number of trial-progress values.');
end
if numberObservations < 3
    error('trialIndexRegressionCV:TooFewObservations', ...
        'At least three observations are required.');
end
if numberAvailableComponents < 1
    error('trialIndexRegressionCV:NoComponents', ...
        'X must contain at least one diffusion component.');
end
if ~isscalar(maxComponents) || maxComponents < 1 || ...
        maxComponents ~= floor(maxComponents)
    error('trialIndexRegressionCV:InvalidMaxComponents', ...
        'maxComponents must be a positive integer.');
end
if any(~isfinite(trialProgress)) || any(trialProgress < 0) || ...
        any(trialProgress > 1)
    error('trialIndexRegressionCV:InvalidTarget', ...
        'Trial progress must contain finite values from zero to one.');
end

effectiveFolds = min(k, numberObservations);
minimumTrainingObservations = numberObservations - ...
    ceil(numberObservations / effectiveFolds);
numberComponents = min([numberAvailableComponents, maxComponents, ...
    max(1, minimumTrainingObservations - 2)]);
componentIndices = 1:numberComponents;
X = X(:, componentIndices);
rng(randomSeed, 'twister');
partition = cvpartition(numberObservations, 'KFold', effectiveFolds);

prediction = NaN(numberObservations, 1);
foldIndex = NaN(numberObservations, 1);
foldR2 = NaN(effectiveFolds, 1);
foldR2Clipped = NaN(effectiveFolds, 1);
foldRMSE = NaN(effectiveFolds, 1);
foldCorrelation = NaN(effectiveFolds, 1);
models = cell(effectiveFolds, 1);
preprocessing = cell(effectiveFolds, 1);

for fold = 1:effectiveFolds
    trainMask = training(partition, fold);
    testMask = test(partition, fold);
    fitted = fitTrialProgressModel( ...
        X(trainMask, :), trialProgress(trainMask));
    foldPrediction = applyTrialProgressModel(fitted, X(testMask, :));

    prediction(testMask) = foldPrediction;
    foldIndex(testMask) = fold;
    [foldR2(fold), foldR2Clipped(fold), foldRMSE(fold), ...
        foldCorrelation(fold)] = regressionMetrics( ...
        trialProgress(testMask), foldPrediction, ...
        mean(trialProgress(trainMask)));
    models{fold} = fitted.model;
    preprocessing{fold} = rmfield(fitted, 'model');
end

[pooledR2, pooledR2Clipped, pooledRMSE, pooledCorrelation] = ...
    regressionMetrics(trialProgress, prediction, mean(trialProgress));

result = struct;
result.mode = "within-run cross-validation";
result.targetDefinition = ...
    "normalized sequential progress within grouped run: (index-1)/(N-1)";
result.targetRange = [0, 1];
result.trueTrialProgress = trialProgress;
result.prediction = prediction;
result.foldIndex = foldIndex;
result.numberFolds = effectiveFolds;
result.foldR2 = foldR2;
result.foldR2Clipped = foldR2Clipped;
result.foldRMSE = foldRMSE;
result.foldCorrelation = foldCorrelation;
result.meanFoldR2 = mean(foldR2, 'omitnan');
result.meanFoldR2Clipped = mean(foldR2Clipped, 'omitnan');
result.meanFoldRMSE = mean(foldRMSE, 'omitnan');
result.meanFoldCorrelation = mean(foldCorrelation, 'omitnan');
result.pooledR2 = pooledR2;
result.pooledR2Clipped = pooledR2Clipped;
result.pooledRMSE = pooledRMSE;
result.pooledCorrelation = pooledCorrelation;
result.models = models;
result.preprocessing = preprocessing;
result.numberAvailableComponents = numberAvailableComponents;
result.numberComponents = numberComponents;
result.componentIndices = componentIndices;
result.minimumTrainingObservations = minimumTrainingObservations;
end


function fitted = fitTrialProgressModel(X, y)
center = mean(X, 1);
scale = std(X, 0, 1);
scale(scale == 0 | ~isfinite(scale)) = 1;
XZ = (X - center) ./ scale;
model = fitlm(XZ, y);
fitted = struct('model', model, 'center', center, 'scale', scale);
end


function prediction = applyTrialProgressModel(fitted, X)
XZ = (X - fitted.center) ./ fitted.scale;
prediction = predict(fitted.model, XZ);
end


function [r2, clippedR2, rmse, correlation] = ...
    regressionMetrics(y, prediction, baselineMean)
residual = y - prediction;
ssResidual = sum(residual .^ 2);
ssTotal = sum((y - baselineMean) .^ 2);
if ssTotal > 0
    r2 = 1 - ssResidual / ssTotal;
else
    r2 = NaN;
end
clippedR2 = max(0, r2);
rmse = sqrt(mean(residual .^ 2));
if numel(y) > 1 && std(y) > 0 && std(prediction) > 0
    correlation = corr(y, prediction, 'Type', 'Pearson');
else
    correlation = NaN;
end
end
