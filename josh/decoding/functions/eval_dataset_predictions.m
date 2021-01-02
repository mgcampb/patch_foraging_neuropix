function [yhat_mean_withinRewsize,yhat_sem_withinRewsize,absError_mean,absError_sem,accuracy] = eval_dataset_predictions(y_true_full,y_hat_full,accuracy_tolerance,dataset_opt)
% eval_dataset_predictions 

    nMice = numel(y_true_full); 
    yhat_mean_withinRewsize = cell(nMice,1); 
    yhat_sem_withinRewsize = cell(nMice,1); 
    absError_mean = cell(nMice,1); 
    absError_sem = cell(nMice,1); 
    accuracy = cell(nMice,1); 

    for mIdx = 1:nMice
        yhat_mean_withinRewsize{mIdx} = cell(numel(dataset_opt.features),1);
        yhat_sem_withinRewsize{mIdx} = cell(numel(dataset_opt.features),1);
        absError_mean{mIdx} = cell(numel(dataset_opt.features),1);
        absError_sem{mIdx} = cell(numel(dataset_opt.features),1);
        accuracy{mIdx} = cell(numel(dataset_opt.features),1);
        for iFeature = 1:numel(dataset_opt.features)
            yhat_mean_withinRewsize{mIdx}{iFeature} = cell(numel(dataset_opt.vars),1);
            yhat_sem_withinRewsize{mIdx}{iFeature} = cell(numel(dataset_opt.vars),1);
            absError_mean{mIdx}{iFeature} = cell(numel(dataset_opt.vars),1);
            absError_sem{mIdx}{iFeature} = cell(numel(dataset_opt.vars),1);
            accuracy{mIdx}{iFeature} = cell(numel(dataset_opt.vars),1);
            for iVar = 1:3
                yhat_mean_withinRewsize{mIdx}{iFeature}{iVar} = cell(numel(dataset_opt.rewsizes),1);
                yhat_sem_withinRewsize{mIdx}{iFeature}{iVar} = cell(numel(dataset_opt.rewsizes),1);
                absError_mean{mIdx}{iFeature}{iVar} = cell(numel(dataset_opt.rewsizes),1);
                absError_sem{mIdx}{iFeature}{iVar} = cell(numel(dataset_opt.rewsizes),1);
                accuracy{mIdx}{iFeature}{iVar} = cell(numel(dataset_opt.rewsizes),1);

                for iRewsize = 1:numel(dataset_opt.rewsizes)
                    i_y_true = y_true_full{mIdx}{iVar}{iRewsize};
                    i_y_hat = y_hat_full{mIdx}{iFeature}{iVar}{iRewsize}; 
                    errors = i_y_true - i_y_hat; 
                    yhat_mean_withinRewsize{mIdx}{iFeature}{iVar}{iRewsize} = nan(max(i_y_true),1);
                    yhat_sem_withinRewsize{mIdx}{iFeature}{iVar}{iRewsize} = nan(max(i_y_true),1);
                    absError_mean{mIdx}{iFeature}{iVar}{iRewsize} = nan(max(i_y_true),1);
                    absError_sem{mIdx}{iFeature}{iVar}{iRewsize} = nan(max(i_y_true),1);
                    accuracy{mIdx}{iFeature}{iVar}{iRewsize} = nan(max(i_y_true),1);
                    for true_label = 1:max(i_y_true) 
                        yhat_mean_withinRewsize{mIdx}{iFeature}{iVar}{iRewsize}(true_label) = mean(i_y_hat(i_y_true == true_label));
                        yhat_sem_withinRewsize{mIdx}{iFeature}{iVar}{iRewsize}(true_label) = std(i_y_hat(i_y_true == true_label)) / sqrt(length(find(i_y_true == true_label)));
                        absError_mean{mIdx}{iFeature}{iVar}{iRewsize}(true_label) = mean(abs(errors(i_y_true == true_label)));
                        absError_sem{mIdx}{iFeature}{iVar}{iRewsize}(true_label) = std(abs(errors(i_y_true == true_label))) / sqrt(length(find(i_y_true == true_label)));
                        accuracy{mIdx}{iFeature}{iVar}{iRewsize}(true_label) = length(find(abs(errors(i_y_true == true_label)) <= accuracy_tolerance)) / length(find(i_y_true == true_label));
                    end
                end
            end
        end
    end

end

