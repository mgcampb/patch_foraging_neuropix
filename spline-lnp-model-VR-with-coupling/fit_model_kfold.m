function [testFit,trainFit,param_mean] = fit_model_kfold(A,spiketrain,test_ind,train_ind)

%% Description
% This code will section the data into 10 different portions. Each portion
% is drawn from across the entire recording session. It will then
% fit the model to 9 sections, and test the model performance on the
% remaining section. This procedure will be repeated 10 times, with all
% possible unique testing sections. The fraction of variance explained, the
% mean-squared error, the log-likelihood increase, and the mean square
% error will be computed for each test data set. In addition, the learned
% parameters will be recorded for each section.


%% Initialize matrices and section the data for k-fold cross-validation

numFolds = length(test_ind);

% initialize matrices
testFit = nan(numFolds,1);
trainFit = nan(numFolds,1); 
numCol = size(A,2);
paramMat = nan(numFolds,numCol);

%% perform k-fold cross validation
for k = 1:numFolds
    fprintf('\t\t- Cross validation fold %d of %d\n', k, numFolds);
    
    test_spikes = spiketrain(test_ind{k}); %test spiking
    test_A = A(test_ind{k},:);
    
    % training data
    train_spikes = spiketrain(train_ind{k});
    train_A = A(train_ind{k},:);
    
    options = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true,'HessianFcn','objective','Display','off');
    
    data{1} = train_A; data{2} = train_spikes;
    if k == 1
        init_param = 1e-3*randn(numCol, 1);
    else
        init_param = param;
    end
    [param] = fminunc(@(param) glm_model(param,data),init_param,options);
    
    % save the parameters
    paramMat(k,:) = param;
    
    %%%%%%%%%%%%% TEST DATA %%%%%%%%%%%%%%%%%%%%%%%
    testFit(k) = compute_llh(test_A,test_spikes,param);
    
    %%%%%%%%%%%%% TRAINING DATA %%%%%%%%%%%%%%%%%%%%%%%
    trainFit(k) = compute_llh(train_A,train_spikes,param);

end

param_mean = nanmean(paramMat);

return
