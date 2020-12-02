paths = struct;
paths.results = 'C:\code\patch_foraging_neuropix\malcolm\glm\GLM_output';
paths.figs = 'C:\figs\patch_foraging_neuropix\glm_plots_for_jan';

session_all = dir(fullfile(paths.results,'*4uL.mat'));
session_all = {session_all.name};
for i = 1:numel(session_all)
    session_all{i} = session_all{i}(1:end-4);
end

var_name = {'Intercept','SessionTime','SessionTime^2','Speed','Speed^2','LickRate',...
    'Kern1','Kern2','Kern3','Kern4','Kern5','Kern6','Kern7','Kern8','Kern9','Kern10','Kern11',...
    'TimeOnPatch','TotalRew','TimeSinceRew'};

for i = 1:numel(session_all)
    session = session_all{i};
    load(fullfile(paths.results,session));
    keep = pval<0.05;
    hfig = figure('Position',[200 200 800 700]);
    hfig.Name = sprintf('%s model coefficients sig only',session);
    boxplot(beta_all(2:end,keep)');
    xticklabels(var_name(2:end));
    set(gca,'TickLabelInterpreter','none');
    ylabel('beta');
    title(session,'Interpreter','none');
    xtickangle(90);

    saveas(hfig,fullfile(paths.figs,hfig.Name),'png');
end