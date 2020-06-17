function [log_llh_test_final,llh_increase_all] = compute_llh_increase(A,spiketrain,variables,final_test_ind)

%% DESCRIPTION: this code will compute a variable-specific llh increase.

% sort the variables
variables = sort(variables);

% compute the test and train indices
final_train_ind = setdiff(1:numel(spiketrain),final_test_ind);
test_spikes = spiketrain(final_test_ind);
train_spikes = spiketrain(final_train_ind);
options = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true,'HessianFcn','objective','Display','off');

%% first, fit the selected model to get a baseline
X = [ones(length(spiketrain),1) cell2mat(A(variables))];
train_A = X(final_train_ind,:);
test_A = X(final_test_ind,:);

% fit the model and compute llh
fprintf('Computing final model fit on last piece of test data \n');
data{1} = train_A; data{2} = train_spikes;
init_param = -1e-2*rand(size(train_A,2), 1);
[param] = fminunc(@(param) glm_model(param,data),init_param,options);
r = exp(test_A * param); n = test_spikes'; meanFR_test = nanmean(test_spikes); 
log_llh_test_baseline = nansum(r-n.*log(r)+log(factorial(n)))/sum(n);
log_llh_test_mean = nansum(meanFR_test-n.*log(meanFR_test)+log(factorial(n)))/sum(n);
log_llh_test_final = (-log_llh_test_baseline + log_llh_test_mean);
log_llh_test_final = log(2)*log_llh_test_final;

%% fit the model on all other variables and compute increase/decrease
num_variables = numel(A);
llh_increase_all = nan(num_variables,1);
for j = 1:4
    
    fprintf('Computing model fit for variable %d\n', j);
    
    if ismember(j,variables)
        % the variable is one of the selected ones... fit the model WITHOUT
        % this one
        X = [ones(length(spiketrain),1) cell2mat(A(setdiff(variables,j)))];
        added = 0;
    else
        % the variable is one of the selected ones... fit the model WITH
        % this one
        X = [ones(length(spiketrain),1) cell2mat(A(union(variables,j)))];
        added = 1;
    end
    
    train_A = X(final_train_ind,:);
    test_A = X(final_test_ind,:);
    
    % fit the model
    data{1} = train_A; data{2} = train_spikes;
    init_param = -1e-2*rand(size(train_A,2), 1);
    [param] = fminunc(@(param) glm_model(param,data),init_param,options);
    
    % compute llh increase from "mean firing rate model"
    r = exp(test_A * param); n = test_spikes';
    
    log_llh_test_model = nansum(r-n.*log(r)+log(factorial(n)))/sum(n);
    if added
        log_llh_test = -log_llh_test_model + log_llh_test_baseline;
    else
        log_llh_test = log_llh_test_model - log_llh_test_baseline;
    end
    log_llh_test = log(2)*log_llh_test;
    
    % fill in all the relevant values for the test fit cases
    llh_increase_all(j) = log_llh_test;
    
end

return