function forwardSearchVis(forward_search,classification_struct,sIdx,session_title,figcounter)
%FORWARDSEARCHVIS Visualize results of forward search on PCs
    precisions = forward_search(sIdx).precisions;
    TP_rates = forward_search(sIdx).TP_rates;
    FP_rates = forward_search(sIdx).FP_rates;   
    ROC_AUC = forward_search(sIdx).ROC_AUC; 
    PR_AUC = forward_search(sIdx).PR_AUC;   
    
    all_concat_labels_noPreRew = horzcat(classification_struct(sIdx).labels_noPreRew{:}) + 1; 
    naive_pr = length(find(all_concat_labels_noPreRew == 1)) / length(all_concat_labels_noPreRew);
    
    precisions_vel = forward_search(sIdx).precisions_vel;
    TP_rates_vel = forward_search(sIdx).TP_rates_vel;
    FP_rates_vel = forward_search(sIdx).FP_rates_vel;
    ROC_AUC_vel = forward_search(sIdx).ROC_AUC_vel;
    PR_AUC_vel = forward_search(sIdx).PR_AUC_vel;
    
    if isnan(forward_search(sIdx).surpass_vel_nPCs) 
        pcs_to_plot = [1,5,10];
    elseif forward_search(sIdx).surpass_vel_nPCs > 3
        pcs_to_plot = [1,2,forward_search(sIdx).surpass_vel_nPCs]; 
    elseif forward_search(sIdx).surpass_vel_nPCs <= 3
        pcs_to_plot = 1:3;
    end 

    curves_legend = {numel(pcs_to_plot)};
    % visualize results with AUROC and Precision-Recall Curve
    for i = 1:numel(pcs_to_plot) 
        pcIdx = pcs_to_plot(i);
        figure(figcounter)
        subplot(1,2,1)
        errorbar(squeeze(mean(FP_rates(pcIdx,:,:))),squeeze(mean(TP_rates(pcIdx,:,:))),1.96 * squeeze(std(TP_rates(pcIdx,:,:))),'linewidth',1.5)
        hold on
        xlabel("Mean False Positive Rate Across Folds")
        ylabel("Mean True Positive Rate Across Folds")
        title(sprintf("%s Receiver Operator Characteristic Curve",session_title))
        subplot(1,2,2)
        errorbar(squeeze(mean(TP_rates(pcIdx,:,:))),squeeze(mean(precisions(pcIdx,:,:))),1.96 * squeeze(std(precisions(pcIdx,:,:))),'linewidth',1.5)
        hold on
        xlabel("Mean True Positive Rate Across Folds")
        ylabel("Mean Precision Across Folds")
        title(sprintf("%s Precision Recall Curve",session_title)) 
        curves_legend{i} = sprintf("Top Decoding PCs 1:%i",pcIdx);
    end  
    
    curves_legend{4} = "Velocity"; 
    curves_legend{5} = "Naive Performance";
    
    % AUC and PR curves
    figure(figcounter)  
    subplot(1,2,1)
    errorbar(mean(FP_rates_vel),mean(TP_rates_vel),1.96 * std(TP_rates_vel),'k','linewidth',1.5)  
    plot([0,1],[0,1],'k--','linewidth',1.5) 
    ylim([0,1])
    legend(curves_legend) 
    subplot(1,2,2) 
    errorbar(mean(TP_rates_vel),mean(precisions_vel),1.96 * std(precisions_vel),'k','linewidth',1.5)
    yline(naive_pr,'k--','linewidth',1.5)
    legend(curves_legend) 
    ylim([0,1]) 
    
    str_decoding_order = num2str(reshape(sprintf('%2d',forward_search(sIdx).pc_decodingOrder),2,[])');
    
    % Now plot AUC increase as we add PCs
    figure(figcounter + 1) 
    subplot(1,2,1)
    errorbar(1:numel(mean(ROC_AUC,2)),mean(ROC_AUC,2),1.96 * std(ROC_AUC,[],2),'linewidth',1.5)
    hold on 
    yline(mean(ROC_AUC_vel),'k--','linewidth',1.5)   
    yline(mean(ROC_AUC_vel) + 1.95 * std(ROC_AUC_vel),'k:','linewidth',1.5) 
    yline(mean(ROC_AUC_vel) - 1.95 * std(ROC_AUC_vel),'k:','linewidth',1.5) 
    title(sprintf("%s AUCROC Forward Search",session_title)) 
    legend("AUCROC for PCs","AUCROC for Velocity") 
    xlabel("PCs Used In Logistic Regression")  
    xticks(1:numel(ROC_AUC_vel))
    xticklabels(str_decoding_order)
    ylabel("AUROC") 
    ylim([0,1])
    subplot(1,2,2)
    errorbar(1:numel(mean(ROC_AUC,2)),mean(PR_AUC,2),1.96 * std(PR_AUC,[],2),'linewidth',1.5) 
    hold on 
    yline(mean(PR_AUC_vel),'k--','linewidth',1.5) 
    yline(mean(PR_AUC_vel) + 1.95 * std(PR_AUC_vel),'k:','linewidth',1.5) 
    yline(mean(PR_AUC_vel) - 1.95 * std(PR_AUC_vel),'k:','linewidth',1.5) 
    title(sprintf("%s AUCPR Forward Search",session_title)) 
    legend("AUCPR for PCs","AUCPR for Velocity") 
    xlabel("PCs Used In Logistic Regression") 
    xticks(1:numel(ROC_AUC_vel))
    xticklabels(str_decoding_order)
    ylabel("AUCPR")  
    ylim([0,1]) 
end

