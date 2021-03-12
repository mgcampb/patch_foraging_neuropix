function [y_true_full,y_hat_full] = predict_dataset_singleSession(X_dataset,y_dataset,models,xval_table,dataset_opt,mouse_names)
    % Predict labels by heldout models
    nMice = numel(X_dataset);
    y_true_full = cell(nMice,1);
    y_hat_full = cell(nMice,1);
    for mIdx = 1:nMice
        y_hat_full{mIdx} = cell(numel(X_dataset{mIdx}),1); 
        y_true_full{mIdx} = cell(numel(X_dataset{mIdx}),1); 
        for i = 1:numel(X_dataset{mIdx})
            y_true_full{mIdx}{i} = cell(numel(dataset_opt.vars),1);
            y_hat_full{mIdx}{i} = cell(numel(dataset_opt.vars),1);
            for iVar = 1:numel(dataset_opt.vars)
                y_true_full{mIdx}{i}{iVar} = cell(numel(dataset_opt.rewsizes),1);
                y_hat_full{mIdx}{i}{iVar} = cell(numel(dataset_opt.rewsizes),1);
                for iRewsize = 1:numel(dataset_opt.rewsizes) 
                    y_true_full{mIdx}{i}{iVar}{iRewsize} = cell(numel(dataset_opt.features{mIdx}{i}),1);

                    % get xval folds for this mouse and reward size 
                    this_rewsize = dataset_opt.rewsizes(iRewsize);
                    foldid = xval_table{mIdx}{i}(xval_table{mIdx}{i}.Rewsize == this_rewsize,:).FoldID; 

                    % get y_true_full  
                    y_true_full_tmp = cell(numel(dataset_opt.numFolds),1); 
                    non_empty_feature = find(cellfun(@(x) ~isempty(x),X_dataset{mIdx}{i}{iVar}{iRewsize}),1);
                    for kFold = 1:dataset_opt.numFolds                        
                        [~,~,~,y_test] = kfold_split_singleSession(X_dataset{mIdx}{i}{iVar}{iRewsize}{non_empty_feature}, ...
                                                                   y_dataset{mIdx}{i}{iVar}{iRewsize}, ...
                                                                   foldid,kFold); 
                        
                        y_true_full_tmp{kFold} = y_test;                                       
                    end
                    y_true_full{mIdx}{i}{iVar}{iRewsize} = cat(1,y_true_full_tmp{:});

                    % now get y_hat for every feature
                    for iFeature = 1:numel(dataset_opt.features{mIdx}{i}) 
                        if ~isempty(X_dataset{mIdx}{i}{iVar}{iRewsize}{iFeature})
                            y_hat_full_tmp = cell(numel(dataset_opt.numFolds),1);
                            for kFold = 1:dataset_opt.numFolds
                                [~,X_test,~,~] = kfold_split_singleSession(X_dataset{mIdx}{i}{iVar}{iRewsize}{iFeature}, ...
                                    y_dataset{mIdx}{i}{iVar}{iRewsize}, ...
                                    foldid,kFold);
                                y_hat = predict(models{mIdx}{i}{iVar}{iRewsize}{iFeature}{kFold},X_test);
                                
                                y_hat_full_tmp{kFold} = y_hat;
                            end
                            y_hat_full{mIdx}{i}{iVar}{iRewsize}{iFeature} = cat(1,y_hat_full_tmp{:});
                        end
                    end
                end
            end 
            fprintf("%s Session %i/%i Complete \n",mouse_names(mIdx),i,numel(X_dataset{mIdx}))
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

