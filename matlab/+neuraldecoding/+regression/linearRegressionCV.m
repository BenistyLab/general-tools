function [meanR2, meanRMSE, meanCorrelation] = linearRegressionCV(X, y, k)
%LINEARREGRESSIONCV Linear regression with k-fold cross-validation.
%
% Inputs:
%   X - N x P feature matrix
%   y - N x 1 continuous target variable
%   k - number of cross-validation folds
%
% Outputs:
%   meanR2          - mean R^2 across folds
%   meanRMSE        - mean root mean squared error across folds
%   meanCorrelation - mean Pearson correlation between predicted and
%                     observed y across folds

y = y(:);

% K-fold cross-validation
cv = cvpartition(length(y), 'KFold', k);

r2 = zeros(k, 1);
rmse = zeros(k, 1);
predictionCorrelation = zeros(k, 1);

for fold = 1:k

    trainIdx = training(cv, fold);
    testIdx = test(cv, fold);

    XTrain = X(trainIdx, :);
    XTest = X(testIdx, :);

    yTrain = y(trainIdx);
    yTest = y(testIdx);

    % -------------------------------------------------------------
    % Standardize features using training data only
    % -------------------------------------------------------------
    mu = mean(XTrain, 1);
    sigma = std(XTrain, [], 1);

    % Avoid division by zero for constant features
    sigma(sigma == 0) = 1;

    XTrain = (XTrain - mu) ./ sigma;
    XTest = (XTest - mu) ./ sigma;

    % -------------------------------------------------------------
    % Fit linear regression on training data
    % -------------------------------------------------------------
    model = fitlm(XTrain, yTrain);

    % -------------------------------------------------------------
    % Predict held-out observations
    % -------------------------------------------------------------
    yPred = predict(model, XTest);

    % -------------------------------------------------------------
    % RMSE
    % -------------------------------------------------------------
    rmse(fold) = sqrt(mean((yTest - yPred).^2));

    % -------------------------------------------------------------
    % Out-of-sample R^2
    %
    % Baseline prediction is the TRAINING-set mean.
    % -------------------------------------------------------------
    ssResidual = sum((yTest - yPred).^2);

    ssTotal = sum((yTest - mean(yTrain)).^2);

    r2(fold) = max(0, 1 - ssResidual / ssTotal);

    % -------------------------------------------------------------
    % Correlation between predicted and actual target
    % -------------------------------------------------------------
    if std(yPred) > 0 && std(yTest) > 0
        predictionCorrelation(fold) = corr( ...
            yPred, ...
            yTest, ...
            'Type', 'Pearson' ...
        );
    else
        predictionCorrelation(fold) = NaN;
    end

end

% -------------------------------------------------------------
% Average across folds
% -------------------------------------------------------------
meanR2 = mean(r2, 'omitnan');
meanRMSE = mean(rmse, 'omitnan');
meanCorrelation = mean(predictionCorrelation, 'omitnan');

end