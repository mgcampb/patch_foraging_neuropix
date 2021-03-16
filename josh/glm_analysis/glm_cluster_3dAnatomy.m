%% set paths
paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice'; 
paths.results = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/glm_results'; 
paths.sig_cells = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/glm_results/gmm/sig_cells_table_gmm_mb_cohort_PFC.mat'; 
paths.figs = '/Users/joshstern/Documents/UchidaLab_NeuralData/neural_data_figs'; % where to save figs
if ~isfolder(paths.figs)
    mkdir(paths.figs);
end
load(paths.sig_cells);
opt.data_set = 'mb';
%% get sessions
session_all = dir(fullfile(paths.results,'*.mat'));
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

%% get significant cells from all sessions
sesh_all = [];
beta_all_sig = [];
cellID_uniq = [];
coordsAP = [];
coordsML = [];
coordsDV = [];
for i = 1:numel(session_all)
    fprintf('Session %d/%d: %s\n',i,numel(session_all),session_all{i});
    
    fit = load(fullfile(paths.results,session_all{i}));
    dat = load(fullfile(paths.data,session_all{i}),'anatomy3d');
%     if ~isfield(dat,'anatomy3d')
%         continue;
%     end

    keep_cell = strcmp(fit.brain_region_rough,opt.brain_region);
    good_cells_this = fit.good_cells(keep_cell);
    
    % sig = fit.pval_full_vs_base(keep_cell)<opt.pval_thresh & sum(abs(fit.beta_all(fit.base_var==0,keep_cell))>0)'>0;
    sig = sum(abs(fit.beta_all(fit.base_var==0,keep_cell))>0)'>0;
    sig_cells = good_cells_this(sig);

    beta_this = fit.beta_all(fit.base_var==0,keep_cell);
    beta_all_sig = [beta_all_sig beta_this(:,sig)];
    sesh_all = [sesh_all; i*ones(sum(sig),1)];
    
    cellID_uniq_this = cell(numel(sig_cells),1);
    for j = 1:numel(sig_cells)
        cellID_uniq_this{j} = sprintf('%s_c%d',fit.opt.session,sig_cells(j));
    end
    cellID_uniq = [cellID_uniq; cellID_uniq_this];
    
    % anatomy coords  
    if isfield(dat(1),'anatomy3d') == true
        keep = ismember(dat.anatomy3d.Coords.CellID,sig_cells);
        assert(all(dat.anatomy3d.Coords.CellID(keep)==sig_cells'));
        coordsAP = [coordsAP; dat.anatomy3d.Coords.AP(keep)];
        coordsML = [coordsML; dat.anatomy3d.Coords.ML(keep)];
        coordsDV = [coordsDV; dat.anatomy3d.Coords.DV(keep)]; 
    else  
        coordsAP = [coordsAP; nan(length(sig_cells),1)];
        coordsML = [coordsML; nan(length(sig_cells),1)];
        coordsDV = [coordsDV; nan(length(sig_cells),1)]; 
    end

end
var_name = fit.var_name(fit.base_var==0)';

%% statistical test anova for differences betw clusters, control by mouse
load(paths.sig_cells);
gmm_idx = sig_cells.GMM_cluster;

% control for mouse
mouse = nan(size(sesh_all));
for i = 1:numel(mouse)
    mouse(i) = str2double(session_all{sesh_all(i)}(1:2));
end

% anova
anova_pval = nan(3,1);
tmp = anovan(coordsAP,{gmm_idx,mouse},'display','off');
anova_pval(1) = tmp(1);

tmp = anovan(coordsML,{gmm_idx,mouse},'display','off');
anova_pval(2) = tmp(1);

tmp = anovan(coordsDV,{gmm_idx,mouse},'display','off');
anova_pval(3) = tmp(1);

%% plot anatomical coords of k means clusters

hfig = figure('Position',[50 50 1500 1250]);
% hfig.Name = sprintf('k means cluster anatomical coords %s cohort %s',opt.data_set,opt.brain_region);
num_clust = length(unique(gmm_idx));
for i = 1:num_clust
    
    subplot(num_clust,3,3*(i-1)+1);
    histogram(coordsAP(gmm_idx==i),10);
    xlim([3000 4000]);
    if i==1
        title(sprintf('AP ANOVA p = %0.3f\nCluster %d',anova_pval(1),i));
    else
        title(sprintf('Cluster %d',i));
    end
    xlabel('AP coord (um)');
    ylabel('num cells');
    set(gca,'FontSize',14);
    
    subplot(num_clust,3,3*(i-1)+2);
    histogram(coordsML(gmm_idx==i),10);
    xlim([4400 5500]);
    if i==1
        title(sprintf('ML ANOVA p = %0.3f\nCluster %d',anova_pval(2),i));
    else
        title(sprintf('Cluster %d',i));
    end
    xlabel('ML coord (um)');
    ylabel('num cells');
    set(gca,'FontSize',14);
    
    subplot(num_clust,3,3*(i-1)+3);
    histogram(coordsDV(gmm_idx==i),10);
    xlim([1500 4500]);
    if i==1
        title(sprintf('DV ANOVA p = %0.3f\nCluster %d',anova_pval(3),i));
    else
        title(sprintf('Cluster %d',i));
    end
    xlabel('DV coord (um)');
    ylabel('num cells');
    set(gca,'FontSize',14);
end

%% Binscatter visualization 
ML_lim = [min(coordsML) max(coordsML)];
DV_lim = [min(coordsDV) max(coordsDV)];
N = 20; 
noise_mag = 30;
figure()
a = scatterhist(noise_mag * randn(size(coordsML)) + coordsML,noise_mag * randn(size(coordsML)) + coordsDV,...
                    'Group',gmm_idx,'Kernel','on',... 
                    'LineStyle',{'-'},...
                    'LineWidth',[2,2,2],...
                    'Marker','.',...
                    'MarkerSize',10);
set(gca, 'YDir','reverse')
set(a(3),'XDir','reverse')
legend("Cluster 1","Cluster 2","Cluster 3","Cluster 4","Cluster 5") 
set(gca,'FontSize',14) 
xlabel("ML Coordinate")
ylabel("DV Coordinate")
% for i_clust = 1:num_clust 
%     subplot(1,num_clust,i_clust)
%     scatterhist(coordsML(gmm_idx==i_clust),coordsDV(gmm_idx==i_clust))
% %     h = binscatter(coordsML(gmm_idx==i_clust),coordsDV(gmm_idx==i_clust),N,'Xlimits',ML_lim,'Ylimits',DV_lim,'ShowEmptyBins','on'); 
% %     imagesc(h.Values)
% end