function [X_dataset,y_dataset,xval_table] = gen_multiclassDataset(X,X_vel,X_pos,X_accel,X_clusters,X_cellIDs,y,y_rewsize,xval_table,mouse_grps,dataset_opt)
    %GEN_MULTICLASSDATASET
    %   Generate xval dataset for multiclass classification problem
    X_dataset = cell(numel(mouse_grps),1);
    y_dataset = cell(numel(mouse_grps),1);
    for mIdx = 1:numel(mouse_grps) % iterate over mice
        X_dataset{mIdx} = cell(numel(dataset_opt.features),1);
        for iFeature = 1:numel(dataset_opt.features)
            X_dataset{mIdx}{iFeature} = cell(numel(dataset_opt.vars),1);
            y_dataset{mIdx} = cell(numel(dataset_opt.vars),1);
            for iVar = 1:numel(dataset_opt.vars) % iterate over variables
                X_dataset{mIdx}{iFeature}{iVar} = cell(numel(dataset_opt.rewsizes),1);
                y_dataset{mIdx}{iVar} = cell(numel(dataset_opt.rewsizes),1);
                for i = 1:numel(mouse_grps{mIdx}) % iterate over sessions, collect data
                    sIdx = mouse_grps{mIdx}(i);
                    rewsize = y_rewsize{mIdx}{i};
                    nTrials = length(rewsize);

                    % Pull out feature
                    if strcmp(dataset_opt.features{iFeature}.type,"KMeans Clusters")
                        neurons_keep = ismember(X_clusters{mIdx}{i},dataset_opt.features{iFeature}.ix); % neuron cluster mask
                        X_session_feature = cellfun(@(x) x(neurons_keep,:),X{mIdx}{i,iVar},'UniformOutput',false); % X w/ neurons of interest 
                    elseif strcmp(dataset_opt.features{iFeature}.type,"CellID")  
                        neurons_keep = ismember(X_cellIDs{mIdx}{i},dataset_opt.features{iFeature}.ix{mIdx}{i}); % neuron cellID mask
                        X_session_feature = cellfun(@(x) x(neurons_keep,:),X{mIdx}{i,iVar},'UniformOutput',false); % X w/ neurons of interest
                    elseif strcmp(dataset_opt.features{iFeature}.type,"Velocity")
                        X_session_feature = X_vel{mIdx}{i,iVar};
                    elseif strcmp(dataset_opt.features{iFeature}.type,"Position")
                        X_session_feature = X_pos{mIdx}{i,iVar};
                    elseif strcmp(dataset_opt.features{iFeature}.type,"Acceleration")
                        X_session_feature = X_accel{mIdx}{i,iVar};
                    end
                    
                    % Shuffle data?
                    if dataset_opt.features{iFeature}.shuffle == true
                        % save indexing so that we can throw this back in trialed form
                        t_lens = cellfun(@(x) size(x,2),X_session_feature);
                        leave_ix = cumsum(t_lens);
                        stop_ix = leave_ix - t_lens + 1;
                        % concatenate, then circshift neurons independently
                        X_session_feature_cat = cat(2,X_session_feature{:});
                        shifts = randi(size(X_session_feature_cat,2),1,size(X_session_feature_cat,1));
                        X_session_feature_cat_shuffle = cell2mat(arrayfun(@(x) circshift(X_session_feature_cat(x,:),[1 shifts(x)]),(1:numel(shifts))','un',0));
                        X_session_feature = arrayfun(@(x) X_session_feature_cat_shuffle(:,stop_ix(x):leave_ix(x)),(1:nTrials)','un',0);
                    end

                    for iRewsize = 1:numel(dataset_opt.rewsizes)
                        this_rewsize = dataset_opt.rewsizes(iRewsize);
                        trials_keep = rewsize == this_rewsize; % rewsize mask  
                        X_dataset{mIdx}{iFeature}{iVar}{iRewsize} = [X_dataset{mIdx}{iFeature}{iVar}{iRewsize};X_session_feature(trials_keep)];  
                        if iFeature == numel(dataset_opt.features)
                            y_dataset{mIdx}{iVar}{iRewsize} = [y_dataset{mIdx}{iVar}{iRewsize};y{mIdx}{i,iVar}(trials_keep)]; 
                        end
                    end
                end
            end
        end

        % Make xval folds, evenly distributing sessions between folds
        for iRewsize = 1:numel(dataset_opt.rewsizes)
            this_rewsize = dataset_opt.rewsizes(iRewsize);
            xval_table_thisRewsize = xval_table{mIdx}(xval_table{mIdx}.Rewsize == this_rewsize,:);
            iRewsize_foldid = nan(size(xval_table_thisRewsize,1),1);
            shift_by = 0; % change which fold is the "last fold" to make sure one fold is not way smaller than the rest
            for i = 1:numel(mouse_grps{mIdx}) % evenly distribute trials from this session between folds
                keep_this = xval_table_thisRewsize.SessionIx == i; % keep trials from this session
                i_nTrials = sum(keep_this); % to ensure proper assignment indexing
                iRewsize_foldid_this = repmat(circshift(1:dataset_opt.numFolds,shift_by),1,ceil(i_nTrials/dataset_opt.numFolds)*dataset_opt.numFolds);
                iRewsize_foldid(keep_this) = iRewsize_foldid_this(1:i_nTrials); % assign folds 1:k
                shift_by = shift_by - mod(i_nTrials,dataset_opt.numFolds); % shift which fold is getting fewer trials
            end
            % assign folds among trials of this reward size
            xval_table{mIdx}(xval_table{mIdx}.Rewsize == this_rewsize,:).FoldID = iRewsize_foldid;
        end
    end
end




