function vis_confusionMat(confusion_mats,X_dataset,session_titles,mutual_information,MSE,var_bins,var_names,vars,rewsizes,vis_features,vis_mice,dataset_opt)
% Visualize confusion matrices, give MI associated per feature in one
% one reward size, one variable to decode
    for i_var = 1:numel(vars) 
        iVar = vars(i_var); 
        n = length(var_bins{iVar})-1; 
        for i_rewsize = 1:numel(rewsizes) 
            iRewsize = rewsizes(i_rewsize); 
            for m_ix = 1:numel(vis_mice)
                mIdx = vis_mice(m_ix); 
                figure()
                for i = 1:numel(X_dataset{mIdx})
                    for i_feature = 1:numel(vis_features) 
                        iFeature = vis_features(i_feature); 
                        subplot(numel(vis_features),numel(X_dataset{mIdx}),i + numel(X_dataset{mIdx}) * (i_feature-1)) 
                        if ~isempty(confusion_mats{mIdx}{i}{iVar}{iRewsize}{iFeature})
                            imagesc(flipud(confusion_mats{mIdx}{i}{iVar}{iRewsize}{iFeature})); %  ./ sum(confusion_mats{mIdx}{i}{iVar}{iRewsize}{iFeature},2)))
                            if iFeature == 1
                                feature_neurons = size(X_dataset{mIdx}{i}{iVar}{iRewsize}{iFeature}{1},1);
                                title(sprintf("%s \n N = %i MI = %.3f MAE = %.3f",session_titles{mIdx}{i},feature_neurons,mutual_information{mIdx}{i}{iVar}{iRewsize}(iFeature)/log(n),MSE{mIdx}{i}{iVar}{iRewsize}(iFeature)))
                            else 
                                feature_neurons = size(X_dataset{mIdx}{i}{iVar}{iRewsize}{iFeature}{1},1);
                                title(sprintf("N = %i MI = %.3f MAE = %.3f",feature_neurons,mutual_information{mIdx}{i}{iVar}{iRewsize}(iFeature)/log(n),MSE{mIdx}{i}{iVar}{iRewsize}(iFeature)))
                            end   
                            if i == 1 % add ylabel and ticks
                                ylabel(sprintf("%s \n True Time (sec)",dataset_opt.features{mIdx}{i}{iFeature}.name))
                                yticks(1:10:(length(var_bins{iVar})-1));
                                yticklabels(fliplr(var_bins{1}(1:10:end)))
                            else
                                yticks([])
                            end 
                            if iFeature == numel(vis_features)   
                                xlabel("Predicted Time (sec)")
                                xticks(1:10:(length(var_bins{iVar})-1));
                                xticklabels(var_bins{1}(1:10:end))
                            else 
                                xticks([])
                            end
                        elseif iFeature == 1
                            title(session_titles{mIdx}{i})
                            xticks([]) 
                        end  
                    end 
                end
                suptitle(sprintf("\n Decoding %s",var_names(iVar)))
            end
        end
    end
end

