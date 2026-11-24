function [accuracy, f1Score, avgActivity, predictedLabels] = ...
    sensorLogisticCV(sensorData, startSample, endSample, labels, numFolds)
% sensorLogisticCV
% Averages sensor activity within a time window and evaluates an
% unregularized binary logistic-regression classifier using cross-validation.
%
% INPUTS
%   sensorData  - Numeric tensor:
%                 sensors x time samples x trials
%   startSample - First sample of the time window, inclusive
%   endSample   - Last sample of the time window, inclusive
%   labels      - One binary label per trial. May be numeric, logical,
%                 string, char, or categorical.
%   numFolds    - Number of cross-validation folds, e.g. 5 or 10
%
% OUTPUTS
%   accuracy        - Overall cross-validated classification accuracy
%   f1Score         - F1 score for the positive class
%   avgActivity     - trials x sensors feature matrix
%   predictedLabels - Cross-validated predictions in the same class order
%                     as categorical(labels)
%
% EXAMPLE
%   [acc, f1, X, yPred] = sensorLogisticCV(data, 101, 200, labels, 5);
%   fprintf('Accuracy: %.3f\nF1 score: %.3f\n', acc, f1);

    arguments
        sensorData (:,:,:) double
        startSample (1,1) double {mustBeInteger, mustBePositive}
        endSample (1,1) double {mustBeInteger, mustBePositive}
        labels
        numFolds (1,1) double {mustBeInteger, mustBeGreaterThanOrEqual(numFolds, 2)} = 5
    end

    % ---------------------------------------------------------------------
    % Validate inputs
    % ---------------------------------------------------------------------
    [numSensors, numTimeSamples, numTrials] = size(sensorData);

    if startSample > endSample
        error('startSample must be less than or equal to endSample.');
    end

    if endSample > numTimeSamples
        error(['endSample (%d) exceeds the number of time samples ' ...
               'in sensorData (%d).'], endSample, numTimeSamples);
    end

    labels = labels(:);

    if numel(labels) ~= numTrials
        error(['The number of labels (%d) must equal the number of ' ...
               'trials (%d).'], numel(labels), numTrials);
    end

    labels = categorical(labels);
    classNames = categories(labels);

    if numel(classNames) ~= 2
        error('Logistic regression requires exactly two label classes.');
    end

    classCounts = countcats(labels);

    if any(classCounts < numFolds)
        error(['Each class must contain at least numFolds observations. ' ...
               'Smallest class has %d observations, but numFolds is %d.'], ...
               min(classCounts), numFolds);
    end

    % ---------------------------------------------------------------------
    % Average activity across the requested time samples
    %
    % Windowed data: sensors x selected samples x trials
    % After averaging: sensors x trials
    % Final features: trials x sensors
    % ---------------------------------------------------------------------
    windowedData = sensorData(:, startSample:endSample, :);

    % "omitnan" averages the available samples when isolated NaNs occur.
    avgActivity = squeeze(mean(windowedData, 2, 'omitnan')).';

    % squeeze can produce an incorrectly oriented vector when there is only
    % one sensor or one trial, so explicitly restore the intended shape.
    avgActivity = reshape(avgActivity, numTrials, numSensors);

    if any(~isfinite(avgActivity), 'all')
        error(['The averaged feature matrix contains NaN or Inf values. ' ...
               'Check for trials/sensors whose entire time window is missing.']);
    end

    % ---------------------------------------------------------------------
    % Stratified K-fold cross-validation
    % ---------------------------------------------------------------------
    rng(1, 'twister');  % Reproducible fold assignments
    cv = cvpartition(labels, 'KFold', numFolds);

    predictedLabels = categorical( ...
        strings(numTrials, 1), classNames, classNames);

    for fold = 1:numFolds
        trainIdx = training(cv, fold);
        testIdx  = test(cv, fold);

        XTrain = avgActivity(trainIdx, :);
        XTest  = avgActivity(testIdx, :);
        yTrain = labels(trainIdx);

        % Convert the two classes to a binary response.
        % The second categorical class is treated as the positive class.
        yTrainBinary = double(yTrain == classNames{2});

        % Fit an unregularized logistic-regression model.
        % fitglm performs maximum-likelihood logistic regression and does
        % not add an L1 or L2 regularization penalty.
        model = fitglm( ...
            XTrain, ...
            yTrainBinary, ...
            'Distribution', 'binomial', ...
            'Link', 'logit');

        % Predicted probability of the positive class.
        positiveProbability = predict(model, XTest);

        % Classify using the conventional probability threshold of 0.5.
        predictedBinary = positiveProbability >= 0.5;

        foldPredictions = repmat(string(classNames{1}), sum(testIdx), 1);
        foldPredictions(predictedBinary) = string(classNames{2});

        predictedLabels(testIdx) = categorical( ...
            foldPredictions, classNames, classNames);
    end

    % ---------------------------------------------------------------------
    % Cross-validated performance metrics
    % ---------------------------------------------------------------------
    positiveClass = classNames{2};

    truePositive  = sum(predictedLabels == positiveClass & ...
                        labels == positiveClass);
    falsePositive = sum(predictedLabels == positiveClass & ...
                        labels ~= positiveClass);
    falseNegative = sum(predictedLabels ~= positiveClass & ...
                        labels == positiveClass);

    accuracy = mean(predictedLabels == labels);

    precisionDenominator = truePositive + falsePositive;
    recallDenominator    = truePositive + falseNegative;

    if precisionDenominator == 0
        precision = 0;
    else
        precision = truePositive / precisionDenominator;
    end

    if recallDenominator == 0
        recall = 0;
    else
        recall = truePositive / recallDenominator;
    end

    if precision + recall == 0
        f1Score = 0;
    else
        f1Score = 2 * precision * recall / (precision + recall);
    end
end