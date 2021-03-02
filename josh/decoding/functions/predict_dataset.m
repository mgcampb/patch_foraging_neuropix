function [confusion_mats,cond_means,cond_sds,y_true_full,y_hat_full] = predict_dataset(X_dataset,y_dataset,models,xval_table,mouse_names,dataset_opt)
% PREDICT_DATASET 
%   Predict labels by heldout models 
    nMice = numel(X_dataset);
    confusion_mats = cell(nMice,1);
    cond_means = cell(nMice,1); 
    cond_sds = cell(nMice,1); 
    y_true_full = cell(nMice,1);
    y_hat_full = cell(nMice,1);
    for mIdx = 1:nMice
        confusion_mats{mIdx} = cell(numel(dataset_opt.features),1);
        cond_means{mIdx} = cell(numel(dataset_opt.features),1);
        cond_sds{mIdx} = cell(numel(dataset_opt.features),1);
        y_hat_full{mIdx} = cell(numel(dataset_opt.features),1);
        for iFeature = 1:numel(dataset_opt.features)
            confusion_mats{mIdx}{iFeature} = cell(numel(dataset_opt.vars),1);
            cond_means{mIdx}{iFeature} = cell(numel(dataset_opt.vars),1);
            cond_sds{mIdx}{iFeature} = cell(numel(dataset_opt.vars),1);
            y_true_full{mIdx} = cell(numel(dataset_opt.vars),1);
            y_hat_full{mIdx}{iFeature} = cell(numel(dataset_opt.vars),1);
            for iVar = 1:3
                confusion_mats{mIdx}{iFeature}{iVar} = cell(numel(dataset_opt.rewsizes),1);
                cond_means{mIdx}{iFeature}{iVar} = cell(numel(dataset_opt.rewsizes),1);
                cond_sds{mIdx}{iFeature}{iVar} = cell(numel(dataset_opt.rewsizes),1);
                y_true_full{mIdx}{iVar} = cell(numel(dataset_opt.rewsizes),1);
                y_hat_full{mIdx}{iFeature}{iVar} = cell(numel(dataset_opt.rewsizes),1);
                for iRewsize = 1:numel(dataset_opt.rewsizes)
                    confusion_mats_tmp = cell(numel(dataset_opt.numFolds),1);
                    cond_means_tmp = cell(numel(dataset_opt.numFolds),1);
                    cond_sds_tmp = cell(numel(dataset_opt.numFolds),1);
                    y_true_full_tmp = cell(numel(dataset_opt.numFolds),1);
                    y_hat_full_tmp = cell(numel(dataset_opt.numFolds),1);
                    this_rewsize = dataset_opt.rewsizes(iRewsize);
                    % xval folds for this mouse and reward size
                    foldid = xval_table{mIdx}(xval_table{mIdx}.Rewsize == this_rewsize,:).FoldID;
                    sessionIx = xval_table{mIdx}(xval_table{mIdx}.Rewsize == this_rewsize,:).SessionIx;
                    for kFold = 1:dataset_opt.numFolds
                        [~,X_test,~,y_test] = kfold_split(X_dataset{mIdx}{iFeature}{iVar}{iRewsize}, ...
                                                          y_dataset{mIdx}{iVar}{iRewsize}, ...
                                                          foldid,kFold,sessionIx);
                        y_hat = predict(models{mIdx}{iFeature}{iVar}{iRewsize}{kFold},X_test');
                        confusion_mats_tmp{kFold} = confusionmat(y_test,y_hat);
                        cond_means_tmp{kFold} = rot90(cellfun(@(x) x(1),models{mIdx}{iFeature}{iVar}{iRewsize}{kFold}.DistributionParameters));
                        cond_sds_tmp{kFold} = rot90(cellfun(@(x) x(2),models{mIdx}{iFeature}{iVar}{iRewsize}{kFold}.DistributionParameters));
                        y_true_full_tmp{kFold} = y_test;
                        y_hat_full_tmp{kFold} = y_hat;
                    end
                    confusion_mats{mIdx}{iFeature}{iVar}{iRewsize} = sum(cat(3,confusion_mats_tmp{:}),3);
                    cond_means{mIdx}{iFeature}{iVar}{iRewsize} = mean(cat(3,cond_means_tmp{:}),3);
                    cond_sds{mIdx}{iFeature}{iVar}{iRewsize} = mean(cat(3,cond_sds_tmp{:}),3);
                    y_true_full{mIdx}{iVar}{iRewsize} = cat(1,y_true_full_tmp{:});
                    y_hat_full{mIdx}{iFeature}{iVar}{iRewsize} = cat(1,y_hat_full_tmp{:});
                end
            end
        end
        if isfield(dataset_opt,'suppressOutput')
            if dataset_opt.suppressOutput == false
                fprintf("%s Model Analysis Complete \n",mouse_names(mIdx))
            end
        else
            fprintf("%s Model Analysis Complete \n",mouse_names(mIdx))
        end
    end
end

