function results = nestedRepeatedL1SVM(X, Y, options)
%NESTEDREPEATEDL1SVM Nested CV for an L1-regularized multiclass linear SVM.
%
% results = nestedRepeatedL1SVM(X,Y)
% results = nestedRepeatedL1SVM(X,Y,options)
%
% This function performs:
%   1. Stratified outer K-fold cross-validation for testing.
%   2. Stratified inner K-fold cross-validation for Lambda selection.
%   3. Repeated random undersampling inside every inner-training fold.
%   4. Selection of Lambda using validation performance averaged over:
%          - inner folds
%          - undersampling repetitions
%   5. Repeated fitting on each complete outer-training partition.
%   6. Majority-vote prediction on the untouched outer-test partition.
%   7. Pooled confusion matrices and classification metrics.
%
% INPUTS
%   X       N-by-P numeric predictor matrix.
%
%   Y       N-by-1 class labels. Y can be numeric, logical, string,
%           categorical, character, or a cell array of character vectors.
%
% OPTIONS
%   options.OuterFolds
%       Number of outer test folds. Default: 5
%
%   options.InnerFolds
%       Number of inner validation folds. Default: 5
%
%   options.InnerRepeats
%       Number of random undersampling repetitions per inner fold.
%       Default: 10
%
%   options.FinalRepeats
%       Number of repeated undersampled models fitted on each complete
%       outer-training partition. Their test predictions are combined by
%       majority voting. Default: 10
%
%   options.LambdaGrid
%       Candidate L1 regularization strengths.
%       Default: logspace(-6,0,20)
%
%   options.SelectionMetric
%       "balancedAccuracy" or "accuracy".
%       Default: "balancedAccuracy"
%
%   options.Coding
%       ECOC coding design. Default: "onevsone"
%
%   options.RandomSeed
%       Base random-number seed. Default: 42
%
%   options.MakePlots
%       Whether to create confusion-matrix and Lambda-selection figures.
%       Default: true
%
% OUTPUT
%   results
%       Structure containing predictions, selected Lambda values, fold
%       metrics, confusion matrices, per-class metrics, and tuning scores.
%
% EXAMPLE
%   options = struct;
%   options.OuterFolds = 5;
%   options.InnerFolds = 5;
%   options.InnerRepeats = 10;
%   options.FinalRepeats = 20;
%   options.LambdaGrid = logspace(-6,0,25);
%   options.SelectionMetric = "balancedAccuracy";
%
%   results = nestedRepeatedL1SVM(X,Y,options);
%
% REQUIREMENT
%   Statistics and Machine Learning Toolbox.

% arguments
%     X double
%     Y
%     options.OuterFolds (1,1) double {mustBeInteger,mustBeGreaterThanOrEqual(options.OuterFolds,2)} = 5
%     options.InnerFolds (1,1) double {mustBeInteger,mustBeGreaterThanOrEqual(options.InnerFolds,2)} = 5
%     options.InnerRepeats (1,1) double {mustBeInteger,mustBePositive} = 10
%     options.FinalRepeats (1,1) double {mustBeInteger,mustBePositive} = 10
%     options.LambdaGrid (1,:) double {mustBePositive} = logspace(-6,0,20)
%     options.SelectionMetric (1,1) string = "balancedAccuracy"
%     options.RandomSeed (1,1) double {mustBeInteger,mustBeNonnegative} = 42
% end

%% Validate and prepare data

if isempty(X)
    error("nestedRepeatedL1SVM:EmptyX", ...
        "X cannot be empty.");
end

Y = categorical(Y);
Y = Y(:);

if size(X,1) ~= numel(Y)
    error("nestedRepeatedL1SVM:SizeMismatch", ...
        "The number of rows in X must equal the number of labels in Y.");
end

if any(~isfinite(X), "all")
    error("nestedRepeatedL1SVM:NonfinitePredictors", ...
        "X contains NaN or Inf values. Impute or remove them first.");
end

if any(isundefined(Y))
    error("nestedRepeatedL1SVM:UndefinedLabels", ...
        "Y contains undefined categorical labels.");
end

% Remove unused categorical levels.
Y = removecats(Y);

classNames = categories(Y);
numberClasses = numel(classNames);
numberObservations = size(X,1);
numberLambdas = numel(options.LambdaGrid);

if numberClasses < 2
    error("nestedRepeatedL1SVM:TooFewClasses", ...
        "At least two classes are required.");
end

validMetrics = ["accuracy","balancedAccuracy"];

if ~any(options.SelectionMetric == validMetrics)
    error("nestedRepeatedL1SVM:InvalidMetric", ...
        "SelectionMetric must be ""accuracy"" or ""balancedAccuracy"".");
end

classOrder = categorical(classNames, classNames);
overallClassCounts = countcats(Y);

if any(overallClassCounts < options.OuterFolds)
    error("nestedRepeatedL1SVM:OuterFoldClassCount", ...
        ["Every class must contain at least OuterFolds observations. " ...
         "Smallest class count: %d; OuterFolds: %d."], ...
         min(overallClassCounts), options.OuterFolds);
end

rng(options.RandomSeed, "twister");

%% Allocate output variables

% Each observation is predicted exactly once as part of an outer-test fold.
pooledPredictions = categorical( ...
    repmat({''}, numberObservations, 1), ...
    classNames);

selectedLambda = NaN(options.OuterFolds,1);
selectedLambdaIndex = NaN(options.OuterFolds,1);

outerAccuracy = NaN(options.OuterFolds,1);
outerBalancedAccuracy = NaN(options.OuterFolds,1);

outerConfusionMatrices = zeros( ...
    numberClasses, numberClasses, options.OuterFolds);

% Mean inner scores after averaging across both inner folds and repetitions.
innerMeanAccuracy = NaN(options.OuterFolds, numberLambdas);
innerMeanBalancedAccuracy = NaN(options.OuterFolds, numberLambdas);

% Standard errors based on all inner-fold/repetition validation scores.
innerSEAccuracy = NaN(options.OuterFolds, numberLambdas);
innerSEBalancedAccuracy = NaN(options.OuterFolds, numberLambdas);

% Store each outer model and its preprocessing parameters.
outerModels = cell(options.OuterFolds, options.FinalRepeats);
outerPreprocessing = cell(options.OuterFolds, options.FinalRepeats);

%% Stratified outer cross-validation

outerPartition = cvpartition( ...
    Y, ...
    KFold=options.OuterFolds);

for outerFold = 1:options.OuterFolds

    fprintf("\n====================================================\n");
    fprintf("Outer fold %d of %d\n", ...
        outerFold, options.OuterFolds);
    fprintf("====================================================\n");

    outerTrainMask = training(outerPartition, outerFold);
    outerTestMask = test(outerPartition, outerFold);

    XOuterTrain = X(outerTrainMask,:);
    YOuterTrain = Y(outerTrainMask);

    XOuterTest = X(outerTestMask,:);
    YOuterTest = Y(outerTestMask);

    %% Verify that inner stratification is possible

    outerTrainCounts = classCountsInOrder( ...
        YOuterTrain, classNames);

    if any(outerTrainCounts < options.InnerFolds)
        error("nestedRepeatedL1SVM:InnerFoldClassCount", ...
            ["Outer fold %d has a class with fewer observations than " ...
             "InnerFolds. Smallest outer-training class count: %d; " ...
             "InnerFolds: %d."], ...
             outerFold, min(outerTrainCounts), options.InnerFolds);
    end

    %% Construct inner stratified folds from outer-training data only

    partitionSeed = options.RandomSeed + 10000 * outerFold;
    rng(partitionSeed, "twister");

    innerPartition = cvpartition( ...
        YOuterTrain, ...
        KFold=options.InnerFolds);

    % Dimensions:
    %   inner fold x undersampling repetition x Lambda
    innerAccuracy = NaN( ...
        options.InnerFolds, ...
        options.InnerRepeats, ...
        numberLambdas);

    innerBalancedAccuracy = NaN( ...
        options.InnerFolds, ...
        options.InnerRepeats, ...
        numberLambdas);

    %% Inner validation loop

    for innerFold = 1:options.InnerFolds

        innerTrainMask = training(innerPartition, innerFold);
        innerValidationMask = test(innerPartition, innerFold);

        XInnerTrainFull = XOuterTrain(innerTrainMask,:);
        YInnerTrainFull = YOuterTrain(innerTrainMask);

        XInnerValidation = XOuterTrain(innerValidationMask,:);
        YInnerValidation = YOuterTrain(innerValidationMask);

        %% Repeated undersampling of this inner-training fold

        for repetition = 1:options.InnerRepeats

            repetitionSeed = ...
                options.RandomSeed + ...
                1000000 * outerFold + ...
                1000 * innerFold + ...
                repetition;

            rng(repetitionSeed, "twister");

            balancedIndices = randomUndersample( ...
                YInnerTrainFull, classNames);

            XInnerTrain = XInnerTrainFull(balancedIndices,:);
            YInnerTrain = YInnerTrainFull(balancedIndices);

            % Estimate preprocessing from this sampled training set only.
            [XInnerTrainZ, center, scale] = ...
                standardizeTrainingData(XInnerTrain);

            XInnerValidationZ = applyStandardization( ...
                XInnerValidation, center, scale);

            % Every Lambda is evaluated using the exact same sampled data.
            for lambdaIndex = 1:numberLambdas

                currentLambda = ...
                    options.LambdaGrid(lambdaIndex);

                model = trainL1ECOCSVM( ...
                    XInnerTrainZ, ...
                    YInnerTrain, ...
                    currentLambda);

                validationPrediction = predict( ...
                    model, XInnerValidationZ);

                innerAccuracy( ...
                    innerFold, repetition, lambdaIndex) = ...
                    mean(validationPrediction == YInnerValidation);

                innerBalancedAccuracy( ...
                    innerFold, repetition, lambdaIndex) = ...
                    balancedAccuracy( ...
                        YInnerValidation, ...
                        validationPrediction, ...
                        classNames);
            end
        end
    end

    %% Average validation performance over inner folds and repetitions

    accuracySamples = reshape( ...
        innerAccuracy, ...
        options.InnerFolds * options.InnerRepeats, ...
        numberLambdas);

    balancedAccuracySamples = reshape( ...
        innerBalancedAccuracy, ...
        options.InnerFolds * options.InnerRepeats, ...
        numberLambdas);

    innerMeanAccuracy(outerFold,:) = ...
        mean(accuracySamples, 1, "omitnan");

    innerMeanBalancedAccuracy(outerFold,:) = ...
        mean(balancedAccuracySamples, 1, "omitnan");

    effectiveAccuracyN = sum(~isnan(accuracySamples), 1);
    effectiveBalancedN = sum(~isnan(balancedAccuracySamples), 1);

    innerSEAccuracy(outerFold,:) = ...
        std(accuracySamples, 0, 1, "omitnan") ./ ...
        sqrt(max(effectiveAccuracyN,1));

    innerSEBalancedAccuracy(outerFold,:) = ...
        std(balancedAccuracySamples, 0, 1, "omitnan") ./ ...
        sqrt(max(effectiveBalancedN,1));

    %% Choose Lambda from averaged inner validation performance

    switch options.SelectionMetric
        case "accuracy"
            lambdaScores = innerMeanAccuracy(outerFold,:);

        case "balancedAccuracy"
            lambdaScores = ...
                innerMeanBalancedAccuracy(outerFold,:);
    end

    bestScore = max(lambdaScores);

    % Resolve exact or numerical ties by taking the largest Lambda,
    % which favors the more strongly regularized model.
    tiedIndices = find( ...
        abs(lambdaScores - bestScore) <= ...
        1e-12 * max(1,abs(bestScore)));

    [~, tiedPosition] = max( ...
        options.LambdaGrid(tiedIndices));

    bestLambdaIndex = tiedIndices(tiedPosition);

    selectedLambdaIndex(outerFold) = bestLambdaIndex;
    selectedLambda(outerFold) = ...
        options.LambdaGrid(bestLambdaIndex);

    fprintf("Selected Lambda: %.6g\n", ...
        selectedLambda(outerFold));

    fprintf("Mean inner validation accuracy: %.4f\n", ...
        innerMeanAccuracy(outerFold,bestLambdaIndex));

    fprintf("Mean inner validation balanced accuracy: %.4f\n", ...
        innerMeanBalancedAccuracy(outerFold,bestLambdaIndex));

    %% Fit repeated models on the complete outer-training partition

    numberOuterTestObservations = size(XOuterTest,1);

    repeatedTestPredictions = categorical( ...
        repmat({''}, ...
        numberOuterTestObservations, ...
        options.FinalRepeats), ...
        classNames);

    for repetition = 1:options.FinalRepeats

        finalSeed = ...
            options.RandomSeed + ...
            100000000 * outerFold + ...
            repetition;

        rng(finalSeed, "twister");

        balancedOuterIndices = randomUndersample( ...
            YOuterTrain, classNames);

        XOuterTrainBalanced = ...
            XOuterTrain(balancedOuterIndices,:);

        YOuterTrainBalanced = ...
            YOuterTrain(balancedOuterIndices);

        [XOuterTrainZ, center, scale] = ...
            standardizeTrainingData(XOuterTrainBalanced);

        XOuterTestZ = applyStandardization( ...
            XOuterTest, center, scale);

        model = trainL1ECOCSVM( ...
            XOuterTrainZ, ...
            YOuterTrainBalanced, ...
            selectedLambda(outerFold));

        repeatedTestPredictions(:,repetition) = ...
            predict(model, XOuterTestZ);

        outerModels{outerFold,repetition} = model;

        outerPreprocessing{outerFold,repetition} = struct( ...
            Center=center, ...
            Scale=scale, ...
            TrainingIndices=find(outerTrainMask), ...
            BalancedLocalIndices=balancedOuterIndices, ...
            RandomSeed=finalSeed);
    end

    %% Combine repeated outer-model predictions by majority vote

    outerFoldPrediction = majorityVote( ...
        repeatedTestPredictions, classNames);

    pooledPredictions(outerTestMask) = ...
        outerFoldPrediction;

    %% Evaluate this outer-test fold

    outerAccuracy(outerFold) = ...
        mean(outerFoldPrediction == YOuterTest);

    outerBalancedAccuracy(outerFold) = ...
        balancedAccuracy( ...
            YOuterTest, ...
            outerFoldPrediction, ...
            classNames);

    outerConfusionMatrices(:,:,outerFold) = ...
        confusionMatrixInOrder( ...
            YOuterTest, ...
            outerFoldPrediction, ...
            classNames);

    fprintf("Outer-test accuracy: %.4f\n", ...
        outerAccuracy(outerFold));

    fprintf("Outer-test balanced accuracy: %.4f\n", ...
        outerBalancedAccuracy(outerFold));
end

%% Pooled nested-cross-validation performance

pooledConfusionMatrix = confusionMatrixInOrder( ...
    Y, pooledPredictions, classNames);

pooledAccuracy = mean(pooledPredictions == Y);

pooledBalancedAccuracy = balancedAccuracy( ...
    Y, pooledPredictions, classNames);

[precision, recall, f1Score, support] = ...
    perClassMetrics(pooledConfusionMatrix);

macroPrecision = mean(precision, "omitnan");
macroRecall = mean(recall, "omitnan");
macroF1 = mean(f1Score, "omitnan");

perClassTable = table( ...
    string(classNames), ...
    support, ...
    precision, ...
    recall, ...
    f1Score, ...
    'VariableNames', ...
    {'Class', 'Support', 'Precision', 'Recall', 'F1Score'});

foldTable = table( ...
    (1:options.OuterFolds)', ...
    selectedLambda, ...
    outerAccuracy, ...
    outerBalancedAccuracy, ...
    'VariableNames', ...
    {'OuterFold', 'SelectedLambda', ...
     'Accuracy', 'BalancedAccuracy'});

%% Print results

fprintf("\n====================================================\n");
fprintf("POOLED NESTED CROSS-VALIDATION RESULTS\n");
fprintf("====================================================\n");

fprintf("Pooled accuracy:          %.4f\n", ...
    pooledAccuracy);

fprintf("Pooled balanced accuracy: %.4f\n", ...
    pooledBalancedAccuracy);

fprintf("Macro precision:          %.4f\n", ...
    macroPrecision);

fprintf("Macro recall:             %.4f\n", ...
    macroRecall);

fprintf("Macro F1 score:           %.4f\n", ...
    macroF1);

fprintf("\nMean outer-fold accuracy: %.4f +/- %.4f\n", ...
    mean(outerAccuracy), std(outerAccuracy));

fprintf("Mean outer-fold balanced accuracy: %.4f +/- %.4f\n", ...
    mean(outerBalancedAccuracy), ...
    std(outerBalancedAccuracy));

fprintf("\nOuter-fold results:\n");
disp(foldTable);

fprintf("Per-class pooled results:\n");
disp(perClassTable);

fprintf("Pooled confusion matrix, rows=true and columns=predicted:\n");
disp(array2table( ...
    pooledConfusionMatrix, ...
    RowNames=classNames, ...
    VariableNames=matlab.lang.makeValidName(classNames)));





%% Assemble output structure

results = struct;

results.Options = options;
results.ClassNames = classNames;

results.Predictions = pooledPredictions;
results.TrueLabels = Y;

results.PooledAccuracy = pooledAccuracy;
results.PooledBalancedAccuracy = pooledBalancedAccuracy;
results.PooledConfusionMatrix = pooledConfusionMatrix;

results.MacroPrecision = macroPrecision;
results.MacroRecall = macroRecall;
results.MacroF1 = macroF1;

results.PerClassMetrics = perClassTable;
results.OuterFoldMetrics = foldTable;

results.SelectedLambda = selectedLambda;
results.SelectedLambdaIndex = selectedLambdaIndex;

results.InnerMeanAccuracy = innerMeanAccuracy;
results.InnerMeanBalancedAccuracy = ...
    innerMeanBalancedAccuracy;

results.InnerSEAccuracy = innerSEAccuracy;
results.InnerSEBalancedAccuracy = ...
    innerSEBalancedAccuracy;

results.OuterAccuracy = outerAccuracy;
results.OuterBalancedAccuracy = ...
    outerBalancedAccuracy;

results.OuterConfusionMatrices = ...
    outerConfusionMatrices;

results.OuterModels = outerModels;
results.OuterPreprocessing = outerPreprocessing;

end


%% ================================================================
% Local helper functions
% ================================================================

function model = trainL1ECOCSVM(X, Y, lambda)
% Fit a multiclass ECOC classifier whose binary learners are
% L1-regularized linear SVMs.

learner = templateLinear( ...
    Learner="svm", ...
    Regularization="lasso", ...
    Lambda=lambda, ...
    Solver="sparsa");

model = fitcecoc( ...
    X, ...
    Y, ...
    Learners=learner);
end


function sampledIndices = randomUndersample(Y, classNames)
% Randomly sample every class down to the size of the smallest class.

classCounts = classCountsInOrder(Y, classNames);
targetCount = min(classCounts);

if targetCount < 1
    error("nestedRepeatedL1SVM:MissingTrainingClass", ...
        "At least one class is absent from a training partition.");
end

numberClasses = numel(classNames);

sampledIndices = zeros( ...
    targetCount * numberClasses, 1);

writeStart = 1;

for classIndex = 1:numberClasses

    classObservationIndices = find( ...
        Y == classNames{classIndex});

    selectedPositions = randperm( ...
        numel(classObservationIndices), ...
        targetCount);

    selectedClassIndices = ...
        classObservationIndices(selectedPositions);

    writeEnd = writeStart + targetCount - 1;

    sampledIndices(writeStart:writeEnd) = ...
        selectedClassIndices;

    writeStart = writeEnd + 1;
end

% Shuffle the combined balanced sample.
sampledIndices = sampledIndices( ...
    randperm(numel(sampledIndices)));
end


function counts = classCountsInOrder(Y, classNames)
% Count observations in an explicitly specified class order.

numberClasses = numel(classNames);
counts = zeros(numberClasses,1);

for classIndex = 1:numberClasses
    counts(classIndex) = sum( ...
        Y == classNames{classIndex});
end
end


function [XZ, center, scale] = standardizeTrainingData(X)
% Calculate standardization parameters using training data only.

center = mean(X,1);
scale = std(X,0,1);

% Constant predictors remain unchanged after centering.
scale(scale == 0 | ~isfinite(scale)) = 1;

XZ = (X - center) ./ scale;
end


function XZ = applyStandardization(X, center, scale)
% Apply previously estimated standardization parameters.

XZ = (X - center) ./ scale;
end


function predictions = majorityVote( ...
    repeatedPredictions, classNames)
% Perform deterministic class-wise majority voting.
%
% In a vote tie, the class occurring first in classNames is selected.

numberObservations = size(repeatedPredictions,1);
numberClasses = numel(classNames);

predictions = categorical( ...
    repmat({''}, numberObservations,1), ...
    classNames);

for observationIndex = 1:numberObservations

    voteCounts = zeros(numberClasses,1);

    for classIndex = 1:numberClasses
        voteCounts(classIndex) = sum( ...
            repeatedPredictions(observationIndex,:) == ...
            classNames{classIndex});
    end

    [~, winningClassIndex] = max(voteCounts);

    predictions(observationIndex) = ...
        classNames{winningClassIndex};
end
end


function value = balancedAccuracy( ...
    trueLabels, predictedLabels, classNames)
% Calculate the unweighted mean recall among represented true classes.

C = confusionMatrixInOrder( ...
    trueLabels, predictedLabels, classNames);

support = sum(C,2);
recall = diag(C) ./ max(support,1);

% A class absent from this particular evaluation fold should not
% contribute a zero recall.
recall(support == 0) = NaN;

value = mean(recall, "omitnan");
end


function C = confusionMatrixInOrder( ...
    trueLabels, predictedLabels, classNames)
% Calculate a confusion matrix with a fixed class order.

numberClasses = numel(classNames);
C = zeros(numberClasses, numberClasses);

for trueClassIndex = 1:numberClasses
    for predictedClassIndex = 1:numberClasses

        C(trueClassIndex,predictedClassIndex) = sum( ...
            trueLabels == classNames{trueClassIndex} & ...
            predictedLabels == ...
            classNames{predictedClassIndex});
    end
end
end


function [precision, recall, f1Score, support] = ...
    perClassMetrics(C)
% Calculate precision, recall, F1, and class support.

truePositive = diag(C);
support = sum(C,2);
predictedCount = sum(C,1)';

precision = truePositive ./ predictedCount;
recall = truePositive ./ support;

precision(predictedCount == 0) = NaN;
recall(support == 0) = NaN;

f1Score = 2 .* precision .* recall ./ ...
    (precision + recall);

f1Score(precision + recall == 0) = 0;
end