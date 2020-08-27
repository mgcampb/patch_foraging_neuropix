paths = struct;

paths.model_fits = 'C:\code\patch_foraging_neuropix\malcolm\glm\GLM_output\allRewSize\no_zscore';
paths.figs = 'C:\figs\patch_foraging_neuropix\glm_time_since_reward_kernels_allRewSize';

model_fits = dir(fullfile(paths.model_fits,'*.mat'));
model_fits = {model_fits.name}';


opt = struct;
opt.tbin = 0.02;
opt.basis_length = 2;
opt.nbasis = 11;
% opt.num_base_var = 6;
% opt.pval_thresh = 0.05;
opt.rew_size = [1 2 4];

%% load all model fits
pval_all = [];
beta_all = [];
for mIdx = 1:numel(model_fits)
    dat = load(fullfile(paths.model_fits,model_fits{mIdx}));
    pval_all = [pval_all; dat.pval];
    beta_all = [beta_all dat.beta_all];
    var_name = dat.var_name;
    bas = dat.bas;
    t_basis = dat.t_basis;
end

%% plot

hfig = figure('Position',[200 200 1400 1000]);
hfig.Name = 'Post reward modulation allRewSize allSessions';

rew_kern = contains(var_name,'Kern') | contains(var_name,'TimeSinceRew');
keep = sum(beta_all(rew_kern,:)>0)>0;
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
caxis([-2.2 2.2]);
cb = colorbar;
xlabel('Time since reward');
ylabel('Neuron, sorted by abs peak');
title('A. Model fit (kernels + ramp): 4uL')
ylabel(cb,'Log firing rate change');

% sort by positive change
subplot(3,3,2);
[~,max_idx] = max(ypred);
[~,sort_idx2] = sort(max_idx);
imagesc(t_basis,1:size(ypred,2),ypred(:,sort_idx2)');
%colormap(bluewhitered(256));
caxis([-2.2 2.2]);
cb = colorbar;
xlabel('Time since reward');
ylabel('Neuron, sorted by pos peak');
title('B. Model fit (kernels + ramp): 4uL')
ylabel(cb,'Log firing rate change');

% sort by negative change
subplot(3,3,3);
[~,min_idx] = min(ypred);
[~,sort_idx3] = sort(min_idx);
imagesc(t_basis,1:size(ypred,2),ypred(:,sort_idx3)');
%colormap(bluewhitered(256));
caxis([-2.2 2.2]);
cb = colorbar;
xlabel('Time since reward');
ylabel('Neuron, sorted by neg peak');
title('C. Model fit (kernels + ramp): 4uL')
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
    caxis([-2.2 2.2]);
    cb = colorbar;
    xlabel('Time since reward');
    ylabel('Neuron, same sort as A');
    title(sprintf('%duL',rew_size_this(rIdx)));
    ylabel(cb,'Log firing rate change');

    % sort by positive change
    subplot(3,3,2+rIdx*3);
    imagesc(t_basis,1:size(ypred,2),ypred(:,sort_idx2)');
    %colormap(bluewhitered(256));
    caxis([-2.2 2.2]);
    cb = colorbar;
    xlabel('Time since reward');
    ylabel('Neuron, same sort as B');
    title(sprintf('%duL',rew_size_this(rIdx)));
    ylabel(cb,'Log firing rate change');

    % sort by negative change
    subplot(3,3,3+rIdx*3);
    imagesc(t_basis,1:size(ypred,2),ypred(:,sort_idx3)');
    caxis([-2.2 2.2]);
    cb = colorbar;
    xlabel('Time since reward');
    ylabel('Neuron, same sort as C');
    title(sprintf('%duL',rew_size_this(rIdx)));
    ylabel(cb,'Log firing rate change');
end
colormap(bluewhitered(256));

save_figs(paths.figs,hfig,'png');