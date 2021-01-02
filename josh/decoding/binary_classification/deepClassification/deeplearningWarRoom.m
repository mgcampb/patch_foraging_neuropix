%% a little script to get familiar with matlab deep learning 

%% Load in the digits dataset 

digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames'); 

%% display some example data 

figure;
perm = randperm(10000,20);
for i = 1:20
    subplot(4,5,i);
    imshow(imds.Files{perm(i)});
end
 
%% start getting set up
% specify the number of classes in the last fully connected layer of your network as the OutputSize argument
labelCount = countEachLabel(imds); 
disp(labelCount) 

% You must specify the size of the images in the input layer of the network
img = readimage(imds,1);
size(img) % 28 x 28

% split training and test data ... these commands are all for image data
numTrainFiles = 750;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize'); 

%% define network architecture ala Keras 
layers = [
    imageInputLayer([28 28 1]) % specify image size
    
    convolution2dLayer(3,8,'Padding','same') % specify some convolution
    batchNormalizationLayer % normalize activations and gradients
    reluLayer % relu nonlinearity
    
    maxPooling2dLayer(2,'Stride',2) % pooling layer
    
    convolution2dLayer(3,16,'Padding','same') % another convolution layer
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(10) % fully connected layer
    softmaxLayer % softmax readout layer
    classificationLayer]; % compute cross entropy loss for class probabilities returned by softmax 

% specify training options
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

%% Train the network
net = trainNetwork(imdsTrain,layers,options); 

% analyze accuracy  
[YPred,probabilities] = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)

%% Time Series Forecasting w/ LSTMs
% To forecast the values of multiple time steps in the future, use the 
% predictAndUpdateState function to predict time steps one at a time and 
% update the network state at each prediction.

% load some chickenpox data
data = chickenpox_dataset;
data = [data{:}];

figure
plot(data)
xlabel("Month")
ylabel("Cases")
title("Monthly Cases of Chickenpox")

% divide training and test data
numTimeStepsTrain = floor(0.9*numel(data));
dataTrain = data(1:numTimeStepsTrain+1);
dataTest = data(numTimeStepsTrain+1:end);

% standardize training data
% note that we must use same normalization for test data
mu = mean(dataTrain);
sig = std(dataTrain);
dataTrainStandardized = (dataTrain - mu) / sig;

% train on values shifted prev one step
XTrain = dataTrainStandardized(1:end-1);
YTrain = dataTrainStandardized(2:end); 

%% define LSTM network 
numFeatures = 1;
numResponses = 1;
numHiddenUnits = 200;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer]; 

options = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress'); 

%% train network 
net = trainNetwork(XTrain,YTrain,layers,options); 

%% forecast future timesteps 

dataTestStandardized = (dataTest - mu) / sig;
XTest = dataTestStandardized(1:end-1); 
% get network state over all training data 
net = predictAndUpdateState(net,XTrain); 
[net,YPred] = predictAndUpdateState(net,YTrain(end)); 

% now make predictions over test data
numTimeStepsTest = numel(XTest);
for i = 2:numTimeStepsTest 
    % here we update state based on previous prediction
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
end 

% unstandardize predictions 
YPred = sig*YPred + mu; 
YTest = dataTest(2:end);
rmse = sqrt(mean((YPred-YTest).^2));  % take rmse 
disp(rmse)

% plot predictions
figure
plot(dataTrain(1:end-1))
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,[data(numTimeStepsTrain) YPred],'.-')
hold off
xlabel("Month")
ylabel("Cases")
title("Forecast")
legend(["Observed" "Forecast"]) 

% plot predictions vs actual data 
figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Cases")
title("Forecast")
subplot(2,1,2)
stem(YPred - YTest)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse) 

%% Now, we get to update test network with actual observations  

net = resetState(net);
net = predictAndUpdateState(net,XTrain); % get up to test point

YPred = [];
numTimeStepsTest = numel(XTest);
for i = 1:numTimeStepsTest 
    % here we're throwing in the observation s.t. we dont build up errors
    [net,YPred(:,i)] = predictAndUpdateState(net,XTest(:,i),'ExecutionEnvironment','cpu');
end 

% unstandardize, return rmse 
YPred = sig*YPred + mu; 
rmse = sqrt(mean((YPred-YTest).^2)); % rmse 
disp(rmse) 

figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Predicted"])
ylabel("Cases")
title("Forecast with Updates")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse)