%% Combined Cycle Power Plant - Linear Regression (MATLAB)
% Predicts net hourly electrical energy output (PE) from:
%   AT - Ambient Temperature (°C)
%   V  - Exhaust Vacuum (cm Hg)
%   AP - Ambient Pressure (mbar)
%   RH - Relative Humidity (%)
%
% Uses 3 sheets for training and 2 sheets for testing.

clear; clc; close all;

dataFile = fullfile('data', 'CCPP', 'Folds5x2_pp.xlsx');
trainSheets = {'Sheet1', 'Sheet2', 'Sheet3'};
testSheets  = {'Sheet4', 'Sheet5'};

%% Load training data (3 sheets)
trainData = [];
for i = 1:numel(trainSheets)
    sheet = readtable(dataFile, 'Sheet', trainSheets{i});
    trainData = [trainData; table2array(sheet)]; %#ok<AGROW>
end

%% Load testing data (2 sheets)
testData = [];
for i = 1:numel(testSheets)
    sheet = readtable(dataFile, 'Sheet', testSheets{i});
    testData = [testData; table2array(sheet)]; %#ok<AGROW>
end

X_train = trainData(:, 1:4);
y_train = trainData(:, 5);
X_test  = testData(:, 1:4);
y_test  = testData(:, 5);

fprintf('Training samples: %d\n', size(X_train, 1));
fprintf('Testing samples:  %d\n\n', size(X_test, 1));

%% Fit linear regression: y = X*beta + beta0
mdl = fitlm(X_train, y_train, ...
    'VarNames', {'AT', 'V', 'AP', 'RH', 'PE'});

disp(mdl);

%% Predict and evaluate
y_train_pred = predict(mdl, X_train);
y_test_pred  = predict(mdl, X_test);

trainRMSE = sqrt(mean((y_train - y_train_pred).^2));
testRMSE  = sqrt(mean((y_test  - y_test_pred).^2));
trainMAE  = mean(abs(y_train - y_train_pred));
testMAE   = mean(abs(y_test  - y_test_pred));
trainR2   = 1 - sum((y_train - y_train_pred).^2) / sum((y_train - mean(y_train)).^2);
testR2    = 1 - sum((y_test  - y_test_pred).^2)  / sum((y_test  - mean(y_test)).^2);

fprintf('\n--- Performance Metrics ---\n');
fprintf('%-20s %10s %10s %10s\n', 'Split', 'MAE', 'RMSE', 'R²');
fprintf('%-20s %10.4f %10.4f %10.4f\n', 'Train (Sheets 1-3)', trainMAE, trainRMSE, trainR2);
fprintf('%-20s %10.4f %10.4f %10.4f\n', 'Test (Sheets 4-5)',  testMAE,  testRMSE,  testR2);

%% Plot: Actual vs Predicted (test set)
figure('Name', 'Actual vs Predicted');
scatter(y_test, y_test_pred, 10, 'filled', 'MarkerFaceAlpha', 0.3);
hold on;
lims = [min([y_test; y_test_pred]), max([y_test; y_test_pred])];
plot(lims, lims, 'r--', 'LineWidth', 1.5);
xlabel('Actual PE (MW)');
ylabel('Predicted PE (MW)');
title('Test Set: Actual vs Predicted Power Output');
legend('Predictions', 'Perfect prediction', 'Location', 'northwest');
grid on;

if ~exist('results', 'dir')
    mkdir('results');
end
saveas(gcf, fullfile('results', 'actual_vs_predicted_matlab.png'));
