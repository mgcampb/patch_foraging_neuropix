function [X_dataset,y_dataset,xval_table] = gen_multiclassDataset_singleSession(X,X_clusters,X_cellIDs,X_vel,X_accel,X_pos,y,y_rewsize,xval_table,dataset_opt,mouse_grps)
%GEN_MULTICLASSDATASET_SINGLESESSION Summary of this function goes here
% make single session multiclass dataset
    X_dataset = cell(numel(mouse_grps),1);
    y_dataset = cell(numel(mouse_grps),1);
    for mIdx = 1:numel(mouse_grps) % iterate over mice
        X_dataset{mIdx} = cell(numel(mouse_grps{mIdx}),1);
        y_dataset{mIdx} = cell(numel(mouse_grps{mIdx}),1);
        for i = 1:numel(mouse_grps{mIdx}) % iterate over sessions
            rewsize = y_rewsize{mIdx}{i};
            X_dataset{mIdx}{i} = cell(numel(dataset_opt.vars),1);
            y_dataset{mIdx}{i} = cell(numel(dataset_opt.vars),1);
            for iVar = 1:numel(dataset_opt.vars) % iterate over variables
                X_dataset{mIdx}{i}{iVar} = cell(numel(dataset_opt.rewsizes),1);
                y_dataset{mIdx}{i}{iVar} = cell(numel(dataset_opt.rewsizes),1);    
                for iRewsize = 1:numel(dataset_opt.rewsizes)  
                    X_dataset{mIdx}{i}{iVar}{iRewsize} = cell(numel(dataset_opt.features{mIdx}{i}),1);
                    y_dataset{mIdx}{i}{iVar}{iRewsize} = cell(numel(dataset_opt.features{mIdx}{i}),1);     
                    this_rewsize = dataset_opt.rewsizes(iRewsize);
                    trials_keep = rewsize == this_rewsize; % rewsize mask  

                    % labels
                    y_dataset{mIdx}{i}{iVar}{iRewsize} = y{mIdx}{i,iVar}(trials_keep);

                    for iFeature = 1:numel(dataset_opt.features{mIdx}{i}) % Pull out feature 
                        if ~isempty(dataset_opt.features{mIdx}{i}{iFeature})
                            if strcmp(dataset_opt.features{mIdx}{i}{iFeature}.type,"KMeans Clusters")
                                neurons_keep = ismember(X_clusters{mIdx}{i},dataset_opt.features{mIdx}{i}{iFeature}.ix); % neuron cluster mask
                                X_session_feature = cellfun(@(x) x(neurons_keep,:),X{mIdx}{i,iVar},'UniformOutput',false); % X w/ neurons of interest
                            elseif strcmp(dataset_opt.features{mIdx}{i}{iFeature}.type,"CellID")
                                neurons_keep = ismember(X_cellIDs{mIdx}{i},dataset_opt.features{mIdx}{i}{iFeature}.ix); % neuron cellID mask
                                X_session_feature = cellfun(@(x) x(neurons_keep,:),X{mIdx}{i,iVar},'UniformOutput',false); % X w/ neurons of interest
                            elseif strcmp(dataset_opt.features{mIdx}{i}{iFeature}.type,"Velocity")
                                X_session_feature = X_vel{mIdx}{i,iVar};
                            elseif strcmp(dataset_opt.features{mIdx}{i}{iFeature}.type,"Position")
                                X_session_feature = X_pos{mIdx}{i,iVar};
                            elseif strcmp(dataset_opt.features{mIdx}{i}{iFeature}.type,"Acceleration")
                                X_session_feature = X_accel{mIdx}{i,iVar};
                            end
                            % features
                            X_dataset{mIdx}{i}{iVar}{iRewsize}{iFeature} = X_session_feature(trials_keep); 
                        end
                    end
                end
            end

            % Make xval folds, evenly distributing sessions between folds
            for iRewsize = 1:numel(dataset_opt.rewsizes)
                this_rewsize = dataset_opt.rewsizes(iRewsize);
                keep_this = xval_table{mIdx}{i}.Rewsize == this_rewsize;
                i_nTrials = sum(keep_this); % to ensure proper assignment indexing
                iRewsize_foldid = repmat(1:dataset_opt.numFolds,1,ceil(i_nTrials/dataset_opt.numFolds)*dataset_opt.numFolds);
                % assign folds among trials of this reward size
                xval_table{mIdx}{i}(xval_table{mIdx}{i}.Rewsize == this_rewsize,:).FoldID = iRewsize_foldid(1:i_nTrials)';
            end
        end
    end
end

