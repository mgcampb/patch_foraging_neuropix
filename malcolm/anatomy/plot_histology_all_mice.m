addpath(genpath('C:\code\patch_foraging_neuropix\malcolm\functions\'));

paths = struct;
paths.data = 'H:\My Drive\UchidaLab\PatchForaging\processed_neuropix_data';
paths.figs = 'C:\figs\patch_foraging_neuropix\anatomy\histology_assignments_all_mice';
if ~isfolder(paths.figs)
    mkdir(paths.figs);
end

opt = struct;
opt.data_set = 'mb';

%%
session_all = dir(fullfile(paths.data,'*.mat'));
session_all = {session_all.name}';
for i = 1:numel(session_all)
    session_all{i} = session_all{i}(1:end-4);
end
if strcmp(opt.data_set,'mc')
    session_all = session_all(contains(session_all,'mc'));
elseif strcmp(opt.data_set,'mb')
    session_all = session_all(~contains(session_all,'mc'));
end

% get mouse name for each session
mouse = cell(size(session_all));
if strcmp(opt.data_set,'mb')
    for i = 1:numel(session_all)
        mouse{i} = session_all{i}(1:2);
    end
elseif strcmp(opt.data_set,'mc')
    for i = 1:numel(session_all)
        mouse{i} = session_all{i}(1:3);
    end
end
uniq_mouse = unique(mouse);

%%
br_all = [];
mouse_all = [];
cortex_all = [];
sesh_num_all = [];
for i = 1:numel(session_all)
    fprintf('Session %d/%d: %s\n',i,numel(session_all),session_all{i});
    
    dat = load(fullfile(paths.data,session_all{i}));
    
    if isfield(dat,'anatomy')
        Ncells = size(dat.anatomy.cell_labels,1);
        br_all = [br_all; dat.anatomy.cell_labels.BrainRegion];
        mouse_all = [mouse_all; repmat(mouse(i),Ncells,1)];
        cortex_all = [cortex_all; dat.anatomy.cell_labels.Cortex];
        sesh_num_all = [sesh_num_all; i*ones(Ncells,1)];
    end

end

%% plot all mice together

[uniq_br,ia,ic] = unique(br_all);
br_counts = accumarray(ic,1);

hfig = figure('Position',[200 200 1000 500]);
hfig.Name = sprintf('brain region all cells %s cohort',opt.data_set);
bar(br_counts);
xticks(1:numel(uniq_br));
xticklabels(uniq_br);
xtickangle(90);
set(gca,'TickLabelInterpreter','none');
set(gca,'FontSize',14);
box off
ylabel('Num. cells');
title(sprintf('%s cohort',opt.data_set));

saveas(hfig,fullfile(paths.figs,hfig.Name),'png');

%% plot individual mice

hfig = figure('Position',[50 50 1000 1200]);
hfig.Name = sprintf('brain region indiv mice %s cohort',opt.data_set);
for i = 1:numel(uniq_mouse)
    keep = strcmp(mouse_all,uniq_mouse{i});
    cell_counts = nan(numel(uniq_br),1);
    for j = 1:numel(uniq_br)
        cell_counts(j) = sum(strcmp(br_all,uniq_br{j}) & keep);
    end
    
    subplot(numel(uniq_mouse),1,i);
    bar(cell_counts);
    xticks(1:numel(uniq_br));
    xticklabels(uniq_br);
    xtickangle(90);
    box off
    set(gca,'TickLabelInterpreter','none');
    set(gca,'FontSize',16);
    ylabel('Num. cells');
    title(sprintf('%s (%d cells from %d sessions)',uniq_mouse{i},sum(keep),numel(unique(sesh_num_all(keep)))));
end

saveas(hfig,fullfile(paths.figs,hfig.Name),'png');

%% plot individual sessions


for i = 1:numel(uniq_mouse)
    
    keep = strcmp(mouse_all,uniq_mouse{i});
    sesh_this = unique(sesh_num_all(keep));
    
    hfig = figure('Position',[25 25 1000 250*numel(sesh_this)]);
    hfig.Name = sprintf('brain region indiv sessions mouse %s',uniq_mouse{i});
    
    for k = 1:numel(sesh_this)
        keep = sesh_num_all==sesh_this(k);

        cell_counts = nan(numel(uniq_br),1);
        for j = 1:numel(uniq_br)
            cell_counts(j) = sum(strcmp(br_all,uniq_br{j}) & keep);
        end

        subplot(numel(sesh_this),1,k);
        bar(cell_counts);
        xticks(1:numel(uniq_br));
        xticklabels(uniq_br);
        xtickangle(90);
        box off
        set(gca,'TickLabelInterpreter','none');
        set(gca,'FontSize',16);
        ylabel('Num. cells');
        title(sprintf('%s (%d cells)',session_all{sesh_this(k)},sum(keep)),'Interpreter','none');
    end
    
    saveas(hfig,fullfile(paths.figs,hfig.Name),'png');
end
