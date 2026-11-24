function [accuracy, predictions, windowInfo, foldIndex, metrics] = ...
    slidingWindowRegressionCV(data, labels, k, windowSize, windowHop)
%SLIDINGWINDOWREGRESSIONCV Sliding-window linear regression with k-fold CV.
%
% Inputs:
%   data       - Numeric tensor with dimensions:
%                sensors x time x trials
%
%   labels     - Numeric vector containing one continuous label per trial.
%
%   k          - Number of cross-validation folds.
%
%   windowSize - Number of time samples included in each window.
%
%   windowHop  - Number of time samples between consecutive window starts.
%
% Outputs:
%   accuracy    - 1 x nWindows vector containing cross-validated R^2
%                 for each window.
%
%   predictions - nTrials x nWindows matrix of out-of-fold predictions.
%
%   windowInfo  - Structure describing the sliding windows:
%                   windowInfo.startSample
%                   windowInfo.endSample
%                   windowInfo.centerSample
%                   windowInfo.windowSize
%                   windowInfo.windowHop
%
%   foldIndex   - nTrials x 1 vector specifying the fold assigned to
%                 each trial.
%
%   metrics     - Structure containing:
%                   metrics.R2
%                   metrics.correlation
%                   metrics.RMSE
%                   metrics.MAE
%
% At each window, every sensor is averaged over time:
%
%   feature(trial, sensor) =
%       mean(data(sensor, windowSamples, trial))
%
% An unregularized ordinary least-squares model is then fitted:
%
%   label = intercept + beta_1*sensor_1 + ... + beta_d*sensor_d

    %% Validate data
    if ~isnumeric(data) || ndims(data) ~= 3
        error(['data must be a numeric 3-D tensor with dimensions ' ...
               'sensors x time x trials.']);
    end

    if ~isnumeric(labels) || ~isvector(labels)
        error('labels must be a numeric vector.');
    end

    if ~isscalar(k) || ~isnumeric(k) || k ~= floor(k) || k < 2
        error('k must be an integer greater than or equal to 2.');
    end

    if ~isscalar(windowSize) || ~isnumeric(windowSize) || ...
            windowSize ~= floor(windowSize) || windowSize < 1
        error('windowSize must be a positive integer.');
    end

    if ~isscalar(windowHop) || ~isnumeric(windowHop) || ...
            windowHop ~= floor(windowHop) || windowHop < 1
        error('windowHop must be a positive integer.');
    end

    [nSensors, nTime, nTrials] = size(data);

    labels = labels(:);

    if numel(labels) ~= nTrials
        error(['The number of labels (%d) must equal the number ' ...
               'of trials (%d).'], numel(labels), nTrials);
    end

    if k > nTrials
        error('k cannot be greater than the number of trials.');
    end

    if windowSize > nTime
        error(['windowSize (%d) cannot be greater than the number ' ...
               'of time samples (%d).'], windowSize, nTime);
    end

    if any(~isfinite(data), 'all')
        error('data contains NaN or Inf values.');
    end

    if any(~isfinite(labels))
        error('labels contains NaN or Inf values.');
    end

    if numel(unique(labels)) < 2
        error('labels must contain at least two different values.');
    end

    %% Define sliding windows
    startSamples = 1:windowHop:(nTime - windowSize + 1);
    endSamples = startSamples + windowSize - 1;
    centerSamples = (startSamples + endSamples) / 2;

    nWindows = numel(startSamples);

    windowInfo.startSample = startSamples;
    windowInfo.endSample = endSamples;
    windowInfo.centerSample = centerSamples;
    windowInfo.windowSize = windowSize;
    windowInfo.windowHop = windowHop;

    %% Create one cross-validation partition
    % The same trial folds are used for all windows, making the accuracy
    % values directly comparable across time.
    cv = cvpartition(nTrials, 'KFold', k);

    foldIndex = zeros(nTrials, 1);

    for fold = 1:k
        foldIndex(test(cv, fold)) = fold;
    end

    %% Allocate predictions
    predictions = nan(nTrials, nWindows);

    %% Train one model per sliding window
    for windowIdx = 1:nWindows

        currentSamples = ...
            startSamples(windowIdx):endSamples(windowIdx);

        % Average each sensor across the samples in the current window.
        %
        % windowAverage initially has dimensions:
        % sensors x 1 x trials
        windowAverage = mean(data(:, currentSamples, :), 2);

        % Convert to:
        % trials x sensors
        X = reshape(windowAverage, nSensors, nTrials).';

        for fold = 1:k
            trainTrials = training(cv, fold);
            testTrials = test(cv, fold);

            XTrain = X(trainTrials, :);
            yTrain = labels(trainTrials);

            XTest = X(testTrials, :);

            % Add an intercept column.
            XTrainWithIntercept = ...
                [ones(sum(trainTrials), 1), XTrain];

            XTestWithIntercept = ...
                [ones(sum(testTrials), 1), XTest];

            % Ordinary least-squares regression with no regularization.
            beta = XTrainWithIntercept \ yTrain;

            % Store predictions only for held-out trials.
            predictions(testTrials, windowIdx) = ...
                XTestWithIntercept * beta;
        end
    end

    %% Calculate cross-validated metrics
    R2 = nan(1, nWindows);
    correlation = nan(1, nWindows);
    RMSE = nan(1, nWindows);
    MAE = nan(1, nWindows);

    totalSumSquares = sum((labels - mean(labels)).^2);

    for windowIdx = 1:nWindows
        predictedLabels = predictions(:, windowIdx);
        residuals = labels - predictedLabels;

        residualSumSquares = sum(residuals.^2);

        R2(windowIdx) = ...
            1 - residualSumSquares / totalSumSquares;

        RMSE(windowIdx) = sqrt(mean(residuals.^2));
        MAE(windowIdx) = mean(abs(residuals));

        if std(predictedLabels) > 0
            correlation(windowIdx) = ...
                corr(labels, predictedLabels, 'Type', 'Pearson');
        end
    end

    accuracy = R2;

    metrics.R2 = R2;
    metrics.correlation = correlation;
    metrics.RMSE = RMSE;
    metrics.MAE = MAE;
end