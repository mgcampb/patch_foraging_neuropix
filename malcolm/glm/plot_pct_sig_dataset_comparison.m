
% for each dataset/brain_region combo:
y1 = pct_sig;
g1 = mouse;


figure('Position',[400 400 300 500]); hold on;

y_all = {y1,y2,y3};
g_all = {g1,g2,g3};

for j = 1:3
    y = y_all{j};
    g = g_all{j};
    uniq_g = unique(g);
    plot_col = lines(numel(uniq_g));
    for i = 1:numel(uniq_g)
        y_this = y(strcmp(g,uniq_g{i}));
        jit = randn(size(y_this))*0.05;
        my_scatter(j*ones(size(y_this))+jit,y_this,plot_col(i,:),0.6);
    end
end
xticks(1:3);
xticklabels({'MB cohort, PFC','MC cohort, PFC','MC cohort, STR'});
xtickangle(90);
xlim([0.5 3.5]);
ylabel('Percent significant');
set(gca,'FontSize',14);