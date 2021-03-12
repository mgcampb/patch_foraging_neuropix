function vis_errorOverTime(vis_mice,vis_var,vis_features,error_mean,error_sem,colors,dataset_opt,session_titles,feature_names,var_bins,var_names,metric_name)
% Visualize fidelity of decoding over time 
    n_rewsizes = numel(dataset_opt.rewsizes); 
    for i_var = 1:numel(vis_var) 
        iVar = vis_var(i_var);
        for m_ix = 1:numel(vis_mice) % 1:numel(abs_error_mean_givenTrue)   
            mIdx = vis_mice(m_ix); 
            nSessions = numel(error_mean{mIdx});
            figure() 
            for i = 1:nSessions
                for iRewsize = 1:n_rewsizes
                    subplot(n_rewsizes,nSessions,(nSessions) * (iRewsize - 1) +  i);hold on 
                    for i_feature = 1:numel(vis_features)
                        iFeature = vis_features(i_feature);
                        if iVar ~= 3
                            shadedErrorBar(var_bins{iVar}(1:end-1),error_mean{mIdx}{i}{iVar}{iRewsize}(iFeature,:),error_sem{mIdx}{i}{iVar}{iRewsize}(iFeature,:),'lineprops',{'Color',colors(iFeature,:)});
                        else
                            shadedErrorBar(-fliplr(var_bins{iVar}(1:end-1)),flipud(error_mean{mIdx}{i}{iVar}{iRewsize}(iFeature,:)),flipud(error_sem{mIdx}{i}{iVar}{iRewsize}(iFeature,:)),'lineprops',{'Color',colors(iFeature,:)});
                        end
                    end
                    if i == 1 
                        ylabel(sprintf("%i uL %s \n %s",dataset_opt.rewsizes(iRewsize),var_names(iVar),metric_name))
                    end
                    if iRewsize == 1 
                        title(sprintf("%s",session_titles{mIdx}{i})) 
                    end 
                    if iRewsize == n_rewsizes
                        xlabel(sprintf("%s (sec)",var_names(iVar))) 
                    end 
                    if iRewsize == 1 && i == 1 
                        legend(feature_names(vis_features)) 
                    end
                end
            end
        end
        suptitle(sprintf("%s Decoding Timecourse",var_names(iVar))) 
    end
end

