function [X_dataset,y_dataset,xval_table] = gen_nb_dataset3(X,X_clusters,X_cellIDs,X_vel,X_accel,X_pos,y,rewsize,xval_table,dataset_opt)
% Generate single session dataset for Naive Bayes decoding, version 3
% make single session multiclass dataset
    X_dataset = cell(numel(X),1);
    y_dataset = cell(numel(X),1);
    for mIdx = 1:numel(X) % iterate over mice
        X_dataset{mIdx} = cell(size(X{mIdx},1),1);
        y_dataset{mIdx} = cell(size(X{mIdx},1),1);
        for i = 1:size(X{mIdx},1) % iterate over sessions
            i_rewsize = rewsize{mIdx}{i};
            X_dataset{mIdx}{i} = cell(2,1);
            y_dataset{mIdx}{i} = cell(numel(dataset_opt.vars),1);
            for iVar = 1:numel(dataset_opt.vars) % iterate over variables
                if iVar == 3 % just use different training data for time until leave
                    iVar_x = 2; 
                else 
                    iVar_x = 1; 
                end
                if iVar == 1 || iVar == 3
                    X_dataset{mIdx}{i}{iVar_x} = cell(numel(dataset_opt.rewsizes),1); 
                end
                y_dataset{mIdx}{i}{iVar} = cell(numel(dataset_opt.rewsizes),1);    
                for iRewsize = 1:numel(dataset_opt.rewsizes)  
                    y_dataset{mIdx}{i}{iVar}{iRewsize} = cell(numel(dataset_opt.features{mIdx}{i}),1);     
                    this_rewsize = dataset_opt.rewsizes(iRewsize);
                    trials_keep = i_rewsize == this_rewsize; % rewsize mask  

                    % labels
                    y_dataset{mIdx}{i}{iVar}{iRewsize} = y{mIdx}{i,iVar}(trials_keep);
                    
                    if iVar == 1 || iVar == 3
                        X_dataset{mIdx}{i}{iVar_x}{iRewsize} = cell(numel(dataset_opt.features{mIdx}{i}),1);
                        for iFeature = 1:numel(dataset_opt.features{mIdx}{i}) % Pull out feature
                            if ~isempty(dataset_opt.features{mIdx}{i}{iFeature})
                                if strcmp(dataset_opt.features{mIdx}{i}{iFeature}.type,"KMeans Clusters")
                                    neurons_keep = ismember(X_clusters{mIdx}{i},dataset_opt.features{mIdx}{i}{iFeature}.ix); % neuron cluster mask
                                    X_session_feature = cellfun(@(x) x(neurons_keep,:),X{mIdx}{i,iVar_x},'un',0); % X w/ neurons of interest
                                elseif strcmp(dataset_opt.features{mIdx}{i}{iFeature}.type,"CellID")
                                    neurons_keep = ismember(X_cellIDs{mIdx}{i},dataset_opt.features{mIdx}{i}{iFeature}.ix); % neuron cellID mask
                                    X_session_feature = cellfun(@(x) x(neurons_keep,:),X{mIdx}{i,iVar_x},'un',0); % X w/ neurons of interest
                                elseif strcmp(dataset_opt.features{mIdx}{i}{iFeature}.type,"Velocity")
                                    X_session_feature = X_vel{mIdx}{i,iVar_x};
                                elseif strcmp(dataset_opt.features{mIdx}{i}{iFeature}.type,"Position")
                                    X_session_feature = X_pos{mIdx}{i,iVar_x};
                                elseif strcmp(dataset_opt.features{mIdx}{i}{iFeature}.type,"Acceleration")
                                    X_session_feature = X_accel{mIdx}{i,iVar_x};
                                end
                                % features
                                X_dataset{mIdx}{i}{iVar_x}{iRewsize}{iFeature} = X_session_feature(trials_keep);
                            end
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

