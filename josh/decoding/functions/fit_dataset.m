function models = fit_dataset(X_dataset,y_dataset,xval_table,mouse_names,dataset_opt)
%FIT_DATASET 
%   Fit multiclass classification problem
    nMice = numel(X_dataset);
    models = cell(nMice,1);
    zero_sigma = 0.5;
    for mIdx = 1:nMice
        models{mIdx} = cell(numel(dataset_opt.features),1);
        % iterate over the variables we are decoding
        for iFeature = 1:numel(dataset_opt.features)
            models{mIdx}{iFeature} = cell(numel(dataset_opt.vars),1);
            for iVar = 1:3
                models{mIdx}{iFeature}{iVar} = cell(numel(dataset_opt.rewsizes),1);
                % iterate over reward sizes of interest
                for iRewsize = 1:numel(dataset_opt.rewsizes)
                    models{mIdx}{iFeature}{iVar}{iRewsize} = cell(dataset_opt.numFolds,1);
                    this_rewsize = dataset_opt.rewsizes(iRewsize);
                    % xval folds for this mouse and reward size
                    foldid = xval_table{mIdx}(xval_table{mIdx}.Rewsize == this_rewsize,:).FoldID;
                    sessionIx = xval_table{mIdx}(xval_table{mIdx}.Rewsize == this_rewsize,:).SessionIx; 
                    % iterate over xval folds and train models
                    for kFold = 1:dataset_opt.numFolds 
                        [X_train,~,y_train,~] = kfold_split(X_dataset{mIdx}{iFeature}{iVar}{iRewsize}, ...
                                                            y_dataset{mIdx}{iVar}{iRewsize}, ...
                                                            foldid,kFold,sessionIx);  
                        % Add some noise s.t. we can avoid zero variance gaussians
                        X_train(X_train == 0) = normrnd(0,zero_sigma,[length(find(X_train == 0)),1]);
                        models{mIdx}{iFeature}{iVar}{iRewsize}{kFold} = fitcnb(X_train',y_train,'Prior','uniform');
                    end
                end
            end
        end 
        if isfield(dataset_opt,'suppressOutput')
            if dataset_opt.suppressOutput == false
                fprintf("%s Model Fitting Complete \n",mouse_names(mIdx))  
            end
        else 
            fprintf("%s Model Fitting Complete \n",mouse_names(mIdx))  
        end
    end

end

