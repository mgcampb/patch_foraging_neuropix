paths = struct;

paths.model_fits = 'C:\data\patch_foraging_neuropix\GLM_output\run_20201110_all_sessions_1SecKerns';
paths.figs = 'C:\figs\patch_foraging_neuropix\glm_time_since_reward_kernels_allRewSize_20201111';

model_fits = dir(fullfile(paths.model_fits,'*.mat'));
model_fits = {model_fits.name}';
% model_fits = model_fits(~contains(model_fits,'mc'));

% model_fits = {'mc2_20201005.mat',...
%     'mc2_20201016.mat',...
%     'mc4_20201025.mat',...
%     'mc4_20201027.mat'};

model_fits = {'mc2_20201020.mat',...
    'mc4_20201024.mat',...
    'mc4_20201026.mat'};

opt = struct;
opt.tbin = 0.02;
opt.basis_length = 2;
opt.nbasis = 11;
% opt.num_base_var = 6;
% opt.pval_thresh = 0.05;
opt.rew_size = [1 2 4];

opt.brain_region = ''; % 'ORB'; % for all, put ''

%% load all model fits
pval_all = [];
beta_all = [];
cell_labels_all = [];
for mIdx = 1:numel(model_fits)
    dat = load(fullfile(paths.model_fits,model_fits{mIdx}));
    pval_all = [pval_all; dat.pval_full_vs_base];
    beta_all = [beta_all dat.beta_all];
    var_name = dat.var_name;
    bas = dat.bas_rew;
    t_basis = dat.t_basis_rew;
    %cell_labels_this = dat.anatomy.cell_labels;
    cell_labels_this = table;
    cell_labels_this.CellID = dat.good_cells_all';
    % cell_labels_this.Cortex = dat.depth_from_surface>-3000;
    cell_labels_this.Cortex = dat.depth_from_surface>-2000; %  & dat.depth_from_surface>-4000;
    cell_labels_all = [cell_labels_all; cell_labels_this(ismember(dat.good_cells_all,dat.good_cells),:)];
end

%% plot

hfig = figure('Position',[200 200 1400 1000]);
hfig.Name = sprintf('Post reward modulation allRewSize allSessions MC cohort MOs %s',opt.brain_region);

rew_kern = contains(var_name,'RewKern') | contains(var_name,'TimeSinceRew');
keep = sum(beta_all(rew_kern,:)>0)>0 & cell_labels_all.Cortex';
beta_filt = beta_all(:,keep);

% Kernels only
which_var = rew_kern & contains(var_name,'4uL');
beta_sig_kern = beta_filt(which_var,:);
x = [bas' t_basis'];
ypred = x*beta_sig_kern; 

% sort by absolute change
subplot(3,3,1);
[~,max_idx] = max(abs(ypred));
[~,sort_idx1] = sort(max_idx);
imagesc(t_basis,1:size(ypred,2),ypred(:,sort_idx1)');
%colormap(bluewhitered(256));
caxis([-1 1]);
cb = colorbar;
xlabel('Time since reward');
ylabel('Neuron');
title(sprintf('A. Model fit (kernels + ramp)\nSorted by abs peak\n4uL'))
ylabel(cb,'Log firing rate change');

% sort by positive change
subplot(3,3,2);
[~,max_idx] = max(ypred);
[~,sort_idx2] = sort(max_idx);
imagesc(t_basis,1:size(ypred,2),ypred(:,sort_idx2)');
%colormap(bluewhitered(256));
caxis([-1 1]);
cb = colorbar;
xlabel('Time since reward');
ylabel('Neuron');
title(sprintf('B. Model fit (kernels + ramp)\nSorted by positive peak\n4uL'))
ylabel(cb,'Log firing rate change');

% sort by negative change
subplot(3,3,3);
[~,min_idx] = min(ypred);
[~,sort_idx3] = sort(min_idx);
imagesc(t_basis,1:size(ypred,2),ypred(:,sort_idx3)');
%colormap(bluewhitered(256));
caxis([-1 1]);
cb = colorbar;
xlabel('Time since reward');
ylabel('Neuron');
title(sprintf('C. Model fit (kernels + ramp)\nSorted by negative peak\n4uL'));
ylabel(cb,'Log firing rate change');

rew_size_this = [2 1];
for rIdx = 1:2
    
    % Kernels only
    which_var = rew_kern & contains(var_name,sprintf('%duL',rew_size_this(rIdx)));
    beta_sig_kern = beta_filt(which_var,:);
    x = [bas' t_basis'];
    ypred = x*beta_sig_kern; 

    % sort by absolute change
    subplot(3,3,1+rIdx*3);
    imagesc(t_basis,1:size(ypred,2),ypred(:,sort_idx1)');
    %colormap(bluewhitered(256));
    caxis([-1 1]);
    cb = colorbar;
    xlabel('Time since reward');
    ylabel('Neuron, same sort as A');
    title(sprintf('%duL',rew_size_this(rIdx)));
    ylabel(cb,'Log firing rate change');

    % sort by positive change
    subplot(3,3,2+rIdx*3);
    imagesc(t_basis,1:size(ypred,2),ypred(:,sort_idx2)');
    %colormap(bluewhitered(256));
    caxis([-1 1]);
    cb = colorbar;
    xlabel('Time since reward');
    ylabel('Neuron, same sort as B');
    title(sprintf('%duL',rew_size_this(rIdx)));
    ylabel(cb,'Log firing rate change');

    % sort by negative change
    subplot(3,3,3+rIdx*3);
    imagesc(t_basis,1:size(ypred,2),ypred(:,sort_idx3)');
    caxis([-1 1]);
    cb = colorbar;
    xlabel('Time since reward');
    ylabel('Neuron, same sort as C');
    title(sprintf('%duL',rew_size_this(rIdx)));
    ylabel(cb,'Log firing rate change');
end
colormap(bluewhitered(256));

save_figs(paths.figs,hfig,'png');