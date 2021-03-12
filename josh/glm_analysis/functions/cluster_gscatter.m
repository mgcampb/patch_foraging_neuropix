function cluster_gscatter(num_vis_pcs,score,labels,n_clust)
    %CLUSTER_GSCATTER Function to plot clustered data across pcs 
    % Visualize gmm results
    figure(); hold on
    for iPC = 2:num_vis_pcs
        if num_vis_pcs > 6
            subplot(2,round(num_vis_pcs/2),iPC-1); hold on; 
        else  
            subplot(1,num_vis_pcs-1,iPC-1); hold on; 
        end
        gscatter(score(:,1),score(:,iPC),labels,lines(n_clust),[],10);
        xlabel('PC1');
        ylabel(sprintf('PC%i',iPC));
        set(gca,'FontSize',14);
        if iPC ~= 1
            hLeg = legend();
            set(hLeg,'visible','off')
        end
    end 
    suptitle("Clusters across PC dimensions") 
end

