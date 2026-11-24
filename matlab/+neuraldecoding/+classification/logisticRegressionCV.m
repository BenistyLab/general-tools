function [meanAccuracy, meanAUC] = logisticRegressionCV(X, y, k)
%LOGISTICREGRESSIONCV Logistic regression with stratified k-fold CV.
%
% Inputs:
%   X - N x P feature matrix
%   y - N x 1 binary labels
%   k - number of cross-validation folds
%
% Outputs:
%   meanAccuracy - mean classification accuracy across folds
%   meanAUC      - mean ROC AUC across folds

y = y(:);

% Convert labels to categorical for fitclinear
yCat = categorical(y);

% Stratified k-fold cross-validation
cv = cvpartition(yCat, ...
    'KFold', k, ...
    'Stratify', true);

accuracy = zeros(k,1);
auc = zeros(k,1);

for fold = 1:k

    trainIdx = training(cv, fold);
    testIdx = test(cv, fold);

    XTrain = X(trainIdx,:);
    XTest = X(testIdx,:);

    yTrain = yCat(trainIdx);
    yTest = yCat(testIdx);

    % Standardize using training data only
    mu = mean(XTrain, 1);
    sigma = std(XTrain, [], 1);

    sigma(sigma == 0) = 1;

    XTrain = (XTrain - mu) ./ sigma;
    XTest = (XTest - mu) ./ sigma;

    % Logistic regression
    model = fitclinear( ...
        XTrain, ...
        yTrain, ...
        'Learner', 'logistic');

    % Predictions and class probabilities
    [yPred, scores] = predict(model, XTest);

    % Accuracy
    accuracy(fold) = mean(yPred == yTest);

    % Identify positive class
    classNames = model.ClassNames;

    % Assume the second class is the positive class
    positiveClass = classNames(2);

    % Probability/score for positive class
    positiveScores = scores(:,2);

    % ROC AUC
    [~,~,~,auc(fold)] = perfcurve( ...
        yTest, ...
        positiveScores, ...
        positiveClass);
end

meanAccuracy = mean(accuracy);
meanAUC = mean(auc);

end