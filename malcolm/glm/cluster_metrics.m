%% Consistency of kmeans clusters

num_iter = 1000;
num_clust = 1:15;

pct_all = nan(numel(num_clust),1);
D_all = nan(numel(num_clust),1);
for ii = 1:numel(num_clust)
    clust_all = nan(size(score,1),num_iter);
    D_this = nan(num_iter,1);
    for iter = 1:num_iter
        [kmeans_idx,~,~,D] = kmeans(score(:,1:2),num_clust(ii));

        % reorder cluster numbers to be consistent across reps
        % order by mean on PC1
        means = nan(opt.num_clust,1);
        for i = 1:opt.num_clust
            means(i) = mean(score(kmeans_idx==i,1));
        end
        [~,sort_idx] = sort(means);
        kmeans_idx2 = nan(size(kmeans_idx));
        for i = 1:opt.num_clust
            kmeans_idx2(kmeans_idx==sort_idx(i)) = i;
        end
        clust_all(:,iter) = kmeans_idx2;
        D_this(iter) = mean(min(D,[],2));
    end

    D_all(ii) = mean(D_this);
    
    pct_clust = nan(size(clust_all,1),1);
    for i = 1:numel(pct_clust)
        pct_clust(i) = sum(clust_all(i,:)==mode(clust_all(i,:)))/num_iter;
    end
    
    pct_all(ii) = mean(pct_clust);
end

% take second derivative
secondDeriv = nan(numel(D_all)-2,1);
for ii = 2:numel(D_all)-1
    secondDeriv(ii-1) = D_all(ii+1)+D_all(ii-1)-2*D_all(ii);
end
[~,max_idx] = max(secondDeriv);
max_idx = max_idx+1;

hfig = figure('Position',[300 300 900 400]);
hfig.Name = sprintf('Picking num clusters %s cohort %s',opt.data_set,opt.brain_region);

subplot(1,2,1);
plot(pct_all,'ko');
ylabel('Avg. fraction in same cluster');
xlabel('Num. clusters');
set(gca,'FontSize',14);
box off;

subplot(1,2,2); hold on;
plot(D_all,'ko');
p = plot([max_idx max_idx],ylim,'b--');
legend(p,'Max. curvature');
ylabel('Avg. distance to nearest centroid');
xlabel('Num. clusters');
set(gca,'FontSize',14);
box off;

saveas(hfig,fullfile(paths.figs,hfig.Name),'png');