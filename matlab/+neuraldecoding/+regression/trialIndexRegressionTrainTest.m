function result = trialIndexRegressionTrainTest( ...
    XTrain, trainTrialProgress, XTest, testTrialProgress, maxComponents)
%TRIALINDEXREGRESSIONTRAINTEST Transfer normalized trial progress.

if nargin < 5 || isempty(maxComponents)
    maxComponents = 10;
end
trainTrialProgress = trainTrialProgress(:);
testTrialProgress = testTrialProgress(:);
if size(XTrain, 1) ~= numel(trainTrialProgress) || ...
        size(XTest, 1) ~= numel(testTrialProgress)
    error('trialIndexRegressionTrainTest:SizeMismatch', ...
        'Feature rows must match their trial-progress vectors.');
end
if size(XTrain, 2) ~= size(XTest, 2)
    error('trialIndexRegressionTrainTest:FeatureMismatch', ...
        'Training and test matrices must contain the same components.');
end
if size(XTrain, 1) < 3 || size(XTrain, 2) < 1
    error('trialIndexRegressionTrainTest:InsufficientTrainingData', ...
        'At least three observations and one component are required.');
end
if ~isscalar(maxComponents) || maxComponents < 1 || ...
        maxComponents ~= floor(maxComponents)
    error('trialIndexRegressionTrainTest:InvalidMaxComponents', ...
        'maxComponents must be a positive integer.');
end
if any(~isfinite(trainTrialProgress)) || ...
        any(~isfinite(testTrialProgress)) || ...
        any(trainTrialProgress < 0 | trainTrialProgress > 1) || ...
        any(testTrialProgress < 0 | testTrialProgress > 1)
    error('trialIndexRegressionTrainTest:InvalidTarget', ...
        'Trial progress must contain finite values from zero to one.');
end

numberAvailableComponents = size(XTrain, 2);
numberComponents = min([numberAvailableComponents, maxComponents, ...
    max(1, size(XTrain, 1) - 2)]);
componentIndices = 1:numberComponents;
XTrain = XTrain(:, componentIndices);
XTest = XTest(:, componentIndices);

center = mean(XTrain, 1);
scale = std(XTrain, 0, 1);
scale(scale == 0 | ~isfinite(scale)) = 1;
XTrainZ = (XTrain - center) ./ scale;
XTestZ = (XTest - center) ./ scale;
model = fitlm(XTrainZ, trainTrialProgress);
prediction = predict(model, XTestZ);

residual = testTrialProgress - prediction;
ssResidual = sum(residual .^ 2);
ssTotal = sum((testTrialProgress - mean(trainTrialProgress)) .^ 2);
if ssTotal > 0
    r2 = 1 - ssResidual / ssTotal;
else
    r2 = NaN;
end
rmse = sqrt(mean(residual .^ 2));
if numel(testTrialProgress) > 1 && std(testTrialProgress) > 0 && ...
        std(prediction) > 0
    predictionCorrelation = corr(testTrialProgress, prediction, ...
        'Type', 'Pearson');
else
    predictionCorrelation = NaN;
end

result = struct;
result.mode = "pre-run model applied to held-out run";
result.targetDefinition = ...
    "normalized sequential progress within grouped run: (index-1)/(N-1)";
result.targetRange = [0, 1];
result.trainTrialProgress = trainTrialProgress;
result.trueTrialProgress = testTrialProgress;
result.prediction = prediction;
result.r2 = r2;
result.r2Clipped = max(0, r2);
result.rmse = rmse;
result.correlation = predictionCorrelation;
result.model = model;
result.preprocessing = struct('center', center, 'scale', scale);
result.numberAvailableComponents = numberAvailableComponents;
result.numberComponents = numberComponents;
result.componentIndices = componentIndices;
end
