function vis_cluster_mean_beta(opt,beta_norm,var_name,kmeans_idx)
%VIS_CLUSTER_MEAN_BETA Summary of this function goes here
        hfig = figure('Position',[50 50 1500 1250]);
    hfig.Name = sprintf('k means avg cluster coefficients %s cohort %s',opt.data_set,opt.brain_region);

    plot_col = cool(3);
    for i = 1:opt.num_clust
        subplot(opt.num_clust,1,i); hold on;
        mean_this = mean(beta_norm(:,kmeans_idx==i),2);
        sem_this = std(beta_norm(:,kmeans_idx==i),[],2)/sqrt(sum(kmeans_idx==i));
        for j = 1:size(mean_this,1)
            if contains(var_name{j},'1uL')
                plot_col_this = plot_col(1,:);
            elseif contains(var_name{j},'2uL')
                plot_col_this = plot_col(2,:);
            elseif contains(var_name{j},'4uL')
                plot_col_this = plot_col(3,:);
            end
            if contains(var_name{j},'RewKern')
                errorbar(j,mean_this(j),sem_this(j),'o','Color',plot_col_this,'MarkerFaceColor',plot_col_this);
            else
                errorbar(j,mean_this(j),sem_this(j),'v','Color',plot_col_this,'MarkerFaceColor',plot_col_this);
            end
        end
        plot(xlim,[0 0],'k:');
        xticks([]);
        ylabel('Coeff');
        title(sprintf('Average of cluster %d (n = %d cells)',i,sum(kmeans_idx==i)));
        set(gca,'FontSize',14);
    end

    xticks(1:numel(var_name));
    xticklabels(var_name);
    xtickangle(90);
    set(gca,'TickLabelInterpreter','none');
    set(gca,'FontSize',14);
%     saveas(hfig,fullfile(paths.figs,hfig.Name),'png');
end

