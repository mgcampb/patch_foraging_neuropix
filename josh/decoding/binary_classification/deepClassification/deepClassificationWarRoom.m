%% Train and evaluate MLP performance on Fisher's iris dataset  
% PR, AUC, etc 

%% First import Fisher Iris dataset and make our neural net
load fisheriris
sp = categorical(species);   
[nSamples,nFeatures] = size(meas); 

propTest = .9;
X_train = meas(1:round(propTest * nSamples),:);  
y_train = sp(1:round(propTest*nSamples)); 
X_test = meas(round(propTest * nSamples):end,:); 
y_test = sp(round(propTest*nSamples):end); 

%% define first architecture... plain old MLP 

layers = [
    featureInputLayer(nFeatures)
    reluLayer % relu nonlinearity 
    
    fullyConnectedLayer(10) % fully connected layer 
    reluLayer 
    
    fullyConnectedLayer(10) % fully connected layer 
    reluLayer
    
    fullyConnectedLayer(3) % fully connected layer
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

%% Train the network and evaluate performance on test set 

net = trainNetwork(X_test,y_train,layers,options);

% analyze accuracy  
[YPred,probabilities] = classify(net,imdsValidation);