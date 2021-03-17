function models = fit_dataset_singleSession(X_dataset,y_dataset,xval_table,dataset_opt,mouse_names)  
    %% Fit naive bayes classifiers to neurons in a single session 
    distribution = 'normal';
    if isfield(dataset_opt,'distribution')
        distribution = dataset_opt.distribution; 
%         disp(distribution)
    end
    
    models = cell(numel(X_dataset),1);
    zero_sigma = 0.5;
    for mIdx = 5 % 1:numel(X_dataset) 
        models{mIdx} = cell(numel(X_dataset{mIdx}),1);
        for i = 2 % 1:numel(X_dataset{mIdx})
            models{mIdx}{i} = cell(numel(dataset_opt.vars),1);
            for iVar = 1:numel(dataset_opt.vars) % iterate over the variables we are decoding
                models{mIdx}{i}{iVar} = cell(numel(dataset_opt.rewsizes),1);
                for iRewsize = 1:numel(dataset_opt.rewsizes) % iterate over reward sizes of interest
                    models{mIdx}{i}{iVar}{iRewsize} = cell(numel(dataset_opt.features{mIdx}{i}),1);
                    % xval folds for this session and reward size
                    this_rewsize = dataset_opt.rewsizes(iRewsize);
                    foldid = xval_table{mIdx}{i}(xval_table{mIdx}{i}.Rewsize == this_rewsize,:).FoldID;

                    for iFeature = 1:numel(dataset_opt.features{mIdx}{i}) % iterate over features  
                        if ~isempty(X_dataset{mIdx}{i}{iVar}{iRewsize}{iFeature})
                            models{mIdx}{i}{iVar}{iRewsize}{iFeature} = cell(dataset_opt.numFolds,1);
                            % iterate over xval folds and train models
                            for kFold = 1:dataset_opt.numFolds
                                [X_train,~,y_train,~] = kfold_split_singleSession(X_dataset{mIdx}{i}{iVar}{iRewsize}{iFeature}, ...
                                    y_dataset{mIdx}{i}{iVar}{iRewsize}, ...
                                    foldid,kFold);

                                % Add some noise s.t. we can avoid zero variance gaussians
                                X_train(X_train == 0) = normrnd(0,zero_sigma,[length(find(X_train == 0)),1]); 
%                                 imagesc(X_train) 
%                                 disp([i,iVar,iRewsize,iFeature])
                                models{mIdx}{i}{iVar}{iRewsize}{iFeature}{kFold} = fitcnb(X_train,y_train,'Prior','uniform','DistributionNames',distribution);
                            end  
                        end
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

