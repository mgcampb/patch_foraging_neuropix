

paths = struct;
paths.results = 'C:\data\patch_foraging_neuropix\GLM_output\run_1Sep2020_PCA_MOs';

paths.figs = 'C:\figs\patch_foraging_neuropix\glm\PCs';
if exist(paths.figs,'dir')~=7
    mkdir(paths.figs);
end

% all sessions to analyze:
session_all = dir(fullfile(paths.results,'*.mat'));
session_all = {session_all.name}';
for i = 1:numel(session_all)
    session_all{i} = session_all{i}(1:end-4);
end

opt = struct;
opt.num_pcs = 6;

all_var = 1:48;
base_var = 1:6;
kern_var = find(contains(var_name,'Kern'));
dec_var = all_var(~ismember(all_var,[base_var kern_var]));

for session_idx = 1:numel(session_all)
    rez = load(fullfile(paths.results,session_all{session_idx}));

    hfig = figure('Position',[200 200 1200 1000]);
    hfig.Name = sprintf('%s brain_region=%s',session_all{session_idx},rez.opt.brain_region);
    for i = 1:opt.num_pcs
        subplot(opt.num_pcs,1,i); hold on;
        scatter(base_var,rez.beta_all(base_var,i),'k');
        scatter(kern_var,rez.beta_all(kern_var,i),'r');
        scatter(dec_var,rez.beta_all(dec_var,i),'b');
        plot(xlim,[0 0],'k:');
        xticks(1:numel(rez.var_name))
        xticklabels('');
        ylabel('beta');
        if i==1
            title(sprintf('%s brain_region=%s\nPC%d %0.1f%%Var',session_all{session_idx},rez.opt.brain_region,i,rez.expl(i)),'Interpreter','none')
        else
            title(sprintf('PC%d %0.1f%%Var',i,rez.expl(i)))
        end
    end
    
    xticklabels(rez.var_name);
    xtickangle(90);
    set(gca,'TickLabelInterpreter','none');
    
    saveas(hfig,fullfile(paths.figs,hfig.Name),'png');
end
