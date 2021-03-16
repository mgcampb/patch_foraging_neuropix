function y_hat_trials = predict_dataset_trialed(trial_decoding_features,X_struct,models,xval_table,dataset_opt,mouse_names,session_titles)
% Create trialed predictions for session based decoders
% Note that this will do x-rewsize decoding! 

    nMice = numel(X_struct.X);
    y_hat_trials = cell(nMice,1);
    for mIdx = 1:numel(y_hat_trials) % iterate over mice 
        nSessions = size(X_struct.X{mIdx},1);
        y_hat_trials{mIdx} = cell(nSessions,1);   
        for i = 1:numel(y_hat_trials{mIdx}) % iterate over sessions 
            session_xval_table = xval_table{mIdx}{i};
            rewsize = session_xval_table.Rewsize; 
            foldID = session_xval_table.FoldID; 
            nTrials = length(rewsize);  

            % get pull out feature data before going into trial loop to save time
            X_session_feature = cell(numel(trial_decoding_features),1); 
            for i_feature = 1:numel(trial_decoding_features) 
                iFeature = trial_decoding_features(i_feature); 
                if ~isempty(dataset_opt.features{mIdx}{i}{iFeature})
                    if strcmp(dataset_opt.features{mIdx}{i}{iFeature}.type,"KMeans Clusters")
                        neurons_keep = ismember(X_struct.X_clusters{mIdx}{i},dataset_opt.features{mIdx}{i}{iFeature}.ix); % neuron cluster mask
                        X_session_feature{i_feature} = cellfun(@(x) x(neurons_keep,:),X_struct.X{mIdx}{i,1},'un',0); % X w/ neurons of interest
                    elseif strcmp(dataset_opt.features{mIdx}{i}{iFeature}.type,"CellID")
                        neurons_keep = ismember(X_struct.X_cellIDs{mIdx}{i},dataset_opt.features{mIdx}{i}{iFeature}.ix); % neuron cellID mask
                        X_session_feature{i_feature} = cellfun(@(x) x(neurons_keep,:),X_struct.X{mIdx}{i,1},'un',0); % X w/ neurons of interest
                    elseif strcmp(dataset_opt.features{mIdx}{i}{iFeature}.type,"Velocity")
                        X_session_feature{i_feature} = X_struct.X_vel{mIdx}{i,1};
                    elseif strcmp(dataset_opt.features{mIdx}{i}{iFeature}.type,"Position")
                        X_session_feature{i_feature} = X_struct.X_pos{mIdx}{i,1};
                    elseif strcmp(dataset_opt.features{mIdx}{i}{iFeature}.type,"Acceleration")
                        X_session_feature{i_feature} = X_struct.X_accel{mIdx}{i,1};
                    end 
                end 
            end

            y_hat_trials{mIdx}{i} = cell(nTrials,1); 
            for iTrial = 1:nTrials % iterate over trials
                trial_fold = foldID(iTrial); % trial fold (if trained_rewsize == rewsize(iTrial)) 
                y_hat_trials{mIdx}{i}{iTrial} = cell(numel(dataset_opt.vars),1); 
                for i_var = 1:numel(dataset_opt.vars) 
                    iVar = dataset_opt.vars(i_var);  
                    y_hat_trials{mIdx}{i}{iTrial}{iVar} = cell(numel(dataset_opt.rewsizes),1); 
                    for trained_rewsize = 1:numel(dataset_opt.rewsizes)
                        y_hat_trials{mIdx}{i}{iTrial}{iVar}{trained_rewsize} = cell(numel(trial_decoding_features),1);
                        for i_feature = 1:numel(trial_decoding_features) 
                            iFeature = trial_decoding_features(i_feature); 
                            if ~isempty(dataset_opt.features{mIdx}{i}{iFeature}) % we have this feature for this session
                                X_trial = X_session_feature{i_feature}{iTrial}; 

                                % trained rewsize == trial rewsize: use model trained on heldout fold
                                if dataset_opt.rewsizes(trained_rewsize) == rewsize(iTrial)
                                    this_model = models{mIdx}{i}{iVar}{trained_rewsize}{iFeature}{trial_fold};
                                    y_hat_trials{mIdx}{i}{iTrial}{iVar}{trained_rewsize}{i_feature} = predict(this_model,X_trial');

                                    % trained rewsize == trial rewsize: avg predictions over 5 folds
                                elseif dataset_opt.rewsizes(trained_rewsize) ~= rewsize(iTrial)
                                    trial_y_hat_tmp = cell(dataset_opt.numFolds,1);
                                    for kFold = 1 % :dataset_opt.numFolds
                                        this_model = models{mIdx}{i}{iVar}{trained_rewsize}{iFeature}{kFold};
                                        trial_y_hat_tmp{kFold} = predict(this_model,X_trial');
                                    end

                                    y_hat_trials{mIdx}{i}{iTrial}{iVar}{trained_rewsize}{i_feature} = mean(cat(2,trial_y_hat_tmp{:}),2);
                                end 
                            end
                        end
                    end
                end 
            end
            fprintf("%s Trial Decoding Complete \n",session_titles{mIdx}{i})
        end 
        fprintf("%s Trial Decoding Complete \n",mouse_names(mIdx))
    end
end

