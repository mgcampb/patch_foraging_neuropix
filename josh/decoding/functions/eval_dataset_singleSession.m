function [cond_means,confusion_mats,mutual_information,MAE, ...
          abs_error_mean_givenTrue,abs_error_mean_givenHat,... 
          abs_error_sem_givenTrue,abs_error_sem_givenHat,...
          accuracy_mean_givenTrue,accuracy_mean_givenHat,...
          accuracy_sem_givenTrue,accuracy_sem_givenHat] = eval_dataset_singleSession(models,y_hat_full,y_true_full,dataset_opt,var_bins)
    nMice = numel(y_hat_full); 
    % Calculate performance metrics on Naive Bayes classification problem 
    cond_means = cell(nMice,1);  
    confusion_mats = cell(nMice,1); 
    mutual_information = cell(nMice,1); 
    MAE = cell(nMice,1); 
    abs_error_mean_givenTrue = cell(nMice,1); 
    abs_error_mean_givenHat = cell(nMice,1); 
    abs_error_sem_givenTrue = cell(nMice,1); 
    abs_error_sem_givenHat = cell(nMice,1); 
    accuracy_mean_givenTrue = cell(nMice,1); 
    accuracy_mean_givenHat = cell(nMice,1); 
    accuracy_sem_givenTrue = cell(nMice,1); 
    accuracy_sem_givenHat = cell(nMice,1); 
    for mIdx = 1:nMice 
        nSessions = numel(y_hat_full{mIdx});
        abs_error_mean_givenTrue{mIdx} = cell(nSessions,1); 
        abs_error_mean_givenHat{mIdx} = cell(nSessions,1); 
        abs_error_sem_givenTrue{mIdx} = cell(nSessions,1); 
        abs_error_sem_givenHat{mIdx} = cell(nSessions,1);  
        accuracy_mean_givenTrue{mIdx} = cell(nSessions,1);  
        accuracy_mean_givenHat{mIdx} = cell(nSessions,1);  
        accuracy_sem_givenTrue{mIdx} = cell(nSessions,1);  
        accuracy_sem_givenHat{mIdx} = cell(nSessions,1);  
        
        cond_means{mIdx} = cell(nSessions,1); 
        confusion_mats{mIdx} = cell(nSessions,1);
        mutual_information{mIdx} = cell(nSessions,1); 
        MAE{mIdx} = cell(nSessions,1); 
        for i = 1:nSessions
            abs_error_mean_givenTrue{mIdx}{i} = cell(numel(dataset_opt.vars),1); 
            abs_error_mean_givenHat{mIdx}{i} = cell(numel(dataset_opt.vars),1); 
            abs_error_sem_givenTrue{mIdx}{i} = cell(numel(dataset_opt.vars),1); 
            abs_error_sem_givenHat{mIdx}{i} = cell(numel(dataset_opt.vars),1); 
            accuracy_mean_givenTrue{mIdx}{i} = cell(numel(dataset_opt.vars),1); 
            accuracy_mean_givenHat{mIdx}{i} = cell(numel(dataset_opt.vars),1); 
            accuracy_sem_givenTrue{mIdx}{i} = cell(numel(dataset_opt.vars),1); 
            accuracy_sem_givenHat{mIdx}{i} = cell(numel(dataset_opt.vars),1); 
            cond_means{mIdx}{i} = cell(numel(dataset_opt.vars),1);  
            confusion_mats{mIdx}{i} = cell(numel(dataset_opt.vars),1);
            mutual_information{mIdx}{i} = cell(numel(dataset_opt.vars),1); 
            MAE{mIdx}{i} = cell(numel(dataset_opt.vars),1); 
            for iVar = 1:numel(dataset_opt.vars)   
                bin_dt = diff(var_bins{iVar}(1:2));
                abs_error_mean_givenTrue{mIdx}{i}{iVar} = cell(numel(dataset_opt.rewsizes),1); 
                abs_error_mean_givenHat{mIdx}{i}{iVar} = cell(numel(dataset_opt.rewsizes),1); 
                abs_error_sem_givenTrue{mIdx}{i}{iVar} = cell(numel(dataset_opt.rewsizes),1); 
                abs_error_sem_givenHat{mIdx}{i}{iVar} = cell(numel(dataset_opt.rewsizes),1); 
                accuracy_mean_givenTrue{mIdx}{i}{iVar} = cell(numel(dataset_opt.rewsizes),1); 
                accuracy_mean_givenTrue{mIdx}{i}{iVar} = cell(numel(dataset_opt.rewsizes),1); 
                accuracy_sem_givenTrue{mIdx}{i}{iVar} = cell(numel(dataset_opt.rewsizes),1); 
                accuracy_sem_givenHat{mIdx}{i}{iVar} = cell(numel(dataset_opt.rewsizes),1); 
                cond_means{mIdx}{i}{iVar} = cell(numel(dataset_opt.rewsizes),1); 
                confusion_mats{mIdx}{i}{iVar} = cell(numel(dataset_opt.rewsizes),1);
                mutual_information{mIdx}{i}{iVar} = cell(numel(dataset_opt.rewsizes),1);
                MAE{mIdx}{i}{iVar} = cell(numel(dataset_opt.rewsizes),1);
                for iRewsize = 1:numel(dataset_opt.rewsizes)    
                    % get y_true and y_hat to eval errors
                    y_true = y_true_full{mIdx}{i}{iVar}{iRewsize};
                    % preallocate metrics 
                    abs_error_mean_givenTrue{mIdx}{i}{iVar}{iRewsize} = nan(numel(dataset_opt.features{mIdx}{i}),length(var_bins{iVar})-1);  
                    abs_error_mean_givenHat{mIdx}{i}{iVar}{iRewsize} = nan(numel(dataset_opt.features{mIdx}{i}),length(var_bins{iVar})-1);  
                    abs_error_sem_givenTrue{mIdx}{i}{iVar}{iRewsize} = nan(numel(dataset_opt.features{mIdx}{i}),length(var_bins{iVar})-1);  
                    abs_error_sem_givenHat{mIdx}{i}{iVar}{iRewsize} = nan(numel(dataset_opt.features{mIdx}{i}),length(var_bins{iVar})-1);  
                    accuracy_mean_givenTrue{mIdx}{i}{iVar}{iRewsize} = nan(numel(dataset_opt.features{mIdx}{i}),length(var_bins{iVar})-1);  
                    accuracy_mean_givenTrue{mIdx}{i}{iVar}{iRewsize} = nan(numel(dataset_opt.features{mIdx}{i}),length(var_bins{iVar})-1);  
                    accuracy_sem_givenTrue{mIdx}{i}{iVar}{iRewsize} = nan(numel(dataset_opt.features{mIdx}{i}),length(var_bins{iVar})-1);  
                    accuracy_sem_givenHat{mIdx}{i}{iVar}{iRewsize} = nan(numel(dataset_opt.features{mIdx}{i}),length(var_bins{iVar})-1);  
                    cond_means{mIdx}{i}{iVar}{iRewsize} = cell(numel(dataset_opt.features{mIdx}{i}),1); 
                    confusion_mats{mIdx}{i}{iVar}{iRewsize} = cell(numel(dataset_opt.features{mIdx}{i}),1); 
                    mutual_information{mIdx}{i}{iVar}{iRewsize} = nan(numel(dataset_opt.features{mIdx}{i}),1); 
                    MAE{mIdx}{i}{iVar}{iRewsize} = nan(numel(dataset_opt.features{mIdx}{i}),1); 
                    for iFeature = 1:(numel(dataset_opt.features{mIdx}{i}))   
                        if ~isempty(dataset_opt.features{mIdx}{i}{iFeature})
                            confusion_mats{mIdx}{i}{iVar}{iRewsize}{iFeature} = confusionmat(y_true_full{mIdx}{i}{iVar}{iRewsize},...
                                y_hat_full{mIdx}{i}{iVar}{iRewsize}{iFeature});
                            mutual_information{mIdx}{i}{iVar}{iRewsize}(iFeature) = MI_confusionmat(confusion_mats{mIdx}{i}{iVar}{iRewsize}{iFeature});

                            y_hat = y_hat_full{mIdx}{i}{iVar}{iRewsize}{iFeature};
                            errors = y_true - y_hat; 
                            MAE{mIdx}{i}{iVar}{iRewsize}(iFeature) = bin_dt * nanmean(abs(errors)); 
                            for this_label = 1:max(y_true) 
                                y_true_ix = y_true == this_label;
                                y_hat_ix = y_hat == this_label;
                                
                                abs_error_mean_givenTrue{mIdx}{i}{iVar}{iRewsize}(iFeature,this_label) = nanmean(abs(errors(y_true_ix)));
                                abs_error_mean_givenHat{mIdx}{i}{iVar}{iRewsize}(iFeature,this_label) = nanmean(abs(errors(y_hat_ix)));  
                                % SEM
                                n_labelTrue = length(find((y_true_ix)));
                                n_labelHat = length(find((y_hat_ix)));
                                abs_error_sem_givenTrue{mIdx}{i}{iVar}{iRewsize}(iFeature,this_label) = 1.95 * nanstd(abs(errors(y_true_ix)))/ sqrt(n_labelTrue);
                                abs_error_sem_givenHat{mIdx}{i}{iVar}{iRewsize}(iFeature,this_label) = 1.95 * nanstd(abs(errors(y_hat_ix))) / sqrt(n_labelHat);
                                
                                accuracy_mean_givenTrue{mIdx}{i}{iVar}{iRewsize}(iFeature,this_label) = mean(y_hat(y_true_ix) == y_true(y_true_ix)); %  / n_labelTrue;
                                accuracy_mean_givenHat{mIdx}{i}{iVar}{iRewsize}(iFeature,this_label) = mean(y_hat(y_hat_ix) == y_true(y_hat_ix)); %  / n_labelHat;
                                accuracy_sem_givenTrue{mIdx}{i}{iVar}{iRewsize}(iFeature,this_label) = std(y_hat(y_true_ix) == 1.95 * y_true(y_true_ix)) / sqrt(n_labelTrue);
                                accuracy_sem_givenHat{mIdx}{i}{iVar}{iRewsize}(iFeature,this_label) = std(y_hat(y_hat_ix) == 1.95 * y_true(y_hat_ix)) / sqrt(n_labelHat);
                            end

                            % Now get conditional means per feature
                            cond_means_tmp = cell(dataset_opt.numFolds,1);
                            for kFold = 1:dataset_opt.numFolds
                                if dataset_opt.distribution == "normal"
                                    cond_means_tmp{kFold} = cellfun(@(x) x(1),models{mIdx}{i}{iVar}{iRewsize}{iFeature}{kFold}.DistributionParameters);
                                else
                                    % get mean of kernel distribution
                                    cond_means_tmp{kFold} = cellfun(@(x) mean(x),models{mIdx}{i}{iVar}{iRewsize}{iFeature}{kFold}.DistributionParameters);
                                end
                            end
                            cond_means{mIdx}{i}{iVar}{iRewsize}{iFeature} = mean(cat(3,cond_means_tmp{:}),3);
                        end
                    end 
                end
            end 
        end 
    end 
end

