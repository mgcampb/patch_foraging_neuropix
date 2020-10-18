%% script to plot neural activity (PC1 and example neuron)
% for large reward patches with just one reward at t=0
% to see if there's a correlation with leave times
% MGC 6/23/2020

% ** TO DO: add firing rate cutoff ** 

paths = struct;
paths.data = 'C:\Users\malcg\Dropbox (Personal)\UchidaLab\processed_neuropix_data';
paths.figs = 'C:\figs\patch_foraging_neuropix\decoding\pca'; % where to save figs

addpath(genpath('C:\code\HGRK_analysis_tools'));
addpath(genpath('C:\code\patch_foraging_neuropix\malcolm\functions'));

% analysis options
opt = struct;
opt.session = '80_20200317'; % session to analyze
opt.tbin = 0.02; % time bin for whole session rate matrix (in sec)
opt.smoothSigma_time = 0.1; % gauss smoothing sigma for rate matrix (in sec)

% for concatenating patches
opt.patch_type = 'R-nil'; % options: 'R-nil' or 'RR-nil'
opt.patch_leave_buffer = 0.5; % in seconds; only takes within patch times up to this amount before patch leave
opt.rew_size = 4;

% ADD: firing rate cutoff (opt.min_fr)

paths.figs = fullfile(paths.figs,opt.session,sprintf('%duL',opt.rew_size),opt.patch_type);

%% load data
dat = load(fullfile(paths.data,opt.session));
good_cells = dat.sp.cids(dat.sp.cgs==2);

% time bins
opt.tstart = 0;
opt.tend = max(dat.sp.st);
tbinedge = opt.tstart:opt.tbin:opt.tend;
tbincent = tbinedge(1:end-1)+opt.tbin/2;

%% extract in-patch times
in_patch = false(size(tbincent));
in_patch_buff = false(size(tbincent)); % add buffer for pca
for i = 1:size(dat.patchCSL,1)
    in_patch(tbincent>=dat.patchCSL(i,2) & tbincent<=dat.patchCSL(i,3)) = true;
    in_patch_buff(tbincent>=dat.patchCSL(i,2) & tbincent<=dat.patchCSL(i,3)-opt.patch_leave_buffer) = true;
end
tbincent = tbincent(in_patch);

%% extract patches of the correct type

% get patch num for each patch
patch_num = nan(size(tbincent));
for i = 1:size(dat.patchCSL,1)
    patch_num(tbincent>=dat.patchCSL(i,2) & tbincent<=dat.patchCSL(i,3)) = i;
end

% reward sequence in each patch
max_prt = ceil(max(dat.patches(:,5)))+1; % add 1 sec buffer
rew_seq = nan(size(dat.patchCSL,1),max_prt+1); % add 1 for zero space
rew_bin_edges = -0.5:1:max_prt+0.5;
for i = 1:size(rew_seq,1)
    rewt_this = dat.rew_ts(dat.rew_ts>dat.patchCSL(i,1) & dat.rew_ts<dat.patchCSL(i,3)+0.5);
    rewt_this = rewt_this-dat.patchCSL(i,2);
    rew_seq(i,:) = histcounts(rewt_this,rew_bin_edges);
end

if strcmp(opt.patch_type,'R-nil')
    patches_to_analyze = find(sum(rew_seq,2)==1 & mod(dat.patches(:,2),10)==opt.rew_size);
elseif strcmp(opt.patch_type,'RR-nil')
    patches_to_analyze = find(sum(rew_seq,2)==2 & rew_seq(:,1)==1 & rew_seq(:,2)==1 & mod(dat.patches(:,2),10)==opt.rew_size);
end
prt = dat.patches(patches_to_analyze,5);

%% PCA on fat firing rate matrix: neurons (N) by time (T) where T is the length of the session

% compute firing rate mat
fr_mat = calcFRVsTime(good_cells,dat,opt);

% take zscore
fr_mat_zscore = my_zscore(fr_mat); % z-score is across whole session including out-of-patch times - is this weird??

% pca on firing rate matrix, only in patches with buffer before patch leave
coeffs = pca(fr_mat_zscore(:,in_patch_buff)');

% project full session onto these PCs
score = coeffs'*fr_mat_zscore;

% only take on-patch times
score = score(:,in_patch)';


%% regress out time in session and time in session^2 to remove slowly varying changes

X = [ones(size(tbincent)); tbincent; tbincent.^2]'; % regressor matrix
beta = X\score;
score_resid = score-X*beta;
beta2 = X\score_resid;

hfig = figure('Position',[200 200 600 1000]);
hfig.Name = sprintf('%s_%duL_%s_PCs1-6 original and residual after removing time and time squared',opt.session,opt.rew_size,opt.patch_type);
counter = 1;
for i = 1:6
    subplot(6,2,counter); hold on;
    counter = counter+1;
    my_scatter(tbincent,score(:,i),'k',0.1);
    plot(tbincent,X*beta(:,i),'r-','LineWidth',2);
    title(sprintf('PC%d: ORIGINAL',i));
    xlabel('time (sec)');
    ylabel('PC');
    
    subplot(6,2,counter); hold on;
    counter = counter+1;
    my_scatter(tbincent,score_resid(:,i),'k',0.1);
    plot(tbincent,X*beta2(:,i),'r-','LineWidth',2);
    title(sprintf('PC%d: RESIDUAL',i));
    xlabel('time (sec)');
    ylabel('PC (residual)');
end

save_figs(paths.figs,hfig,'png');

%% plot pc traces for these patches: ORIGINAL

T = ceil(max(prt));
t = 0:opt.tbin:T;
N = numel(t);

hfig = figure('Position',[100 300 2100 800]); hold on;
hfig.Name = sprintf('%s_PCs1-6 - %d uL patches - %s',opt.session,opt.rew_size,opt.patch_type);

% plot PRTs vs patch number in session
subplot(2,7,1); hold on;
plot_col = winter(numel(patches_to_analyze));
[prt_sorted,sort_idx] = sort(prt);
patches_to_analyze_sorted = patches_to_analyze(sort_idx);
for i = 1:numel(patches_to_analyze)
    my_scatter(patches_to_analyze_sorted(i),prt_sorted(i),plot_col(i,:),0.5);
end
ylabel('PRT (sec)');
xlabel('Patch Num');
axis square;

% plot PCs
for pc_num = 1:6
    pca_selected_patches = nan(numel(patches_to_analyze),N);
    for i = 1:numel(patches_to_analyze)
        pca_this = score(patch_num==patches_to_analyze(i),pc_num);
        pca_selected_patches(i,1:numel(pca_this)) = pca_this;
    end

    % sort by prt
    [~,sort_idx] = sort(prt);
    pca_selected_patches_sort = pca_selected_patches(sort_idx,:);

    % plot
    subplot(2,7,pc_num+1); hold on;
    plot_col = winter(numel(patches_to_analyze));
    for i = 1:numel(patches_to_analyze)
        y_this = pca_selected_patches_sort(i,:);
        lh = plot(t,y_this);
        lh.Color = [plot_col(i,:) 0.8];
        max_idx = find(~isnan(y_this),1,'last');
        my_scatter(t(max_idx),y_this(max_idx),plot_col(i,:),0.5);
    end
    xlabel('Time on patch (sec)');
    ylabel('PC')
    title(sprintf('PC%d',pc_num));
    plot([min(prt) min(prt)],ylim,'k--');
    axis square;

    % plot significance
    for i = 1:size(pca_selected_patches,2)
        keep = ~isnan(pca_selected_patches(:,i));
        y_this = pca_selected_patches(keep,i);
        prt_this = prt(keep);
        if numel(y_this)>1
            [r,p] = corrcoef(y_this,prt_this);
            if p(1,2)<0.05
                plot(t(i),max(ylim)-0.5,'k.');
            end
        end
    end
    
    % plot rewards
    if strcmp(opt.patch_type,'R-nil')
        plot(0,max(ylim),'bv','MarkerFaceColor','b');
    elseif strcmp(opt.patch_type,'RR-nil')
        plot(0,max(ylim),'bv','MarkerFaceColor','b');
        plot(1,max(ylim),'bv','MarkerFaceColor','b');
    end
       
end

% SAME but colored by patch number (proxy for time in session)
% plot PRTs vs patch number in session
subplot(2,7,8); hold on;
plot_col = autumn(numel(patches_to_analyze));
for i = 1:numel(patches_to_analyze)
    my_scatter(patches_to_analyze(i),prt(i),plot_col(i,:),0.5);
end
ylabel('PRT (sec)');
xlabel('Patch Num');
axis square;

% plot PCs
for pc_num = 1:6
    pca_selected_patches = nan(numel(patches_to_analyze),N);
    for i = 1:numel(patches_to_analyze)
        pca_this = score(patch_num==patches_to_analyze(i),pc_num);
        pca_selected_patches(i,1:numel(pca_this)) = pca_this;
    end

    % plot
    subplot(2,7,7+pc_num+1); hold on;
    plot_col = autumn(numel(patches_to_analyze));
    for i = 1:numel(patches_to_analyze)
        y_this = pca_selected_patches(i,:);
        lh = plot(t,y_this);
        lh.Color = [plot_col(i,:) 0.8];
        max_idx = find(~isnan(y_this),1,'last');
        my_scatter(t(max_idx),y_this(max_idx),plot_col(i,:),0.5);
    end
    xlabel('Time on patch (sec)');
    ylabel('PC')
    title(sprintf('PC%d',pc_num));
    plot([min(prt) min(prt)],ylim,'k--');
    axis square;

    % plot significance
    for i = 1:size(pca_selected_patches,2)
        keep = ~isnan(pca_selected_patches(:,i));
        y_this = pca_selected_patches(keep,i);
        x_this = patches_to_analyze(keep);
        if numel(y_this)>1
            [r,p] = corrcoef(y_this,x_this);
            if p(1,2)<0.05
                plot(t(i),max(ylim)-0.5,'k.');
            end
        end
    end
    
    % plot rewards
    if strcmp(opt.patch_type,'R-nil')
        plot(0,max(ylim),'bv','MarkerFaceColor','b');
    elseif strcmp(opt.patch_type,'RR-nil')
        plot(0,max(ylim),'bv','MarkerFaceColor','b');
        plot(1,max(ylim),'bv','MarkerFaceColor','b');
    end
       
end

save_figs(paths.figs,hfig,'png');

%% plot pc traces for these patches: TIME AND TIME^2 REGRESSED OUT

T = ceil(max(prt));
t = 0:opt.tbin:T;
N = numel(t);

hfig = figure('Position',[100 300 2100 800]); hold on;
hfig.Name = sprintf('%s_PCs1-6 - %d uL patches - %s - regressed out time in session and time squared',opt.session,opt.rew_size,opt.patch_type);

% plot PRTs vs patch number in session
subplot(2,7,1); hold on;
plot_col = winter(numel(patches_to_analyze));
[prt_sorted,sort_idx] = sort(prt);
patches_to_analyze_sorted = patches_to_analyze(sort_idx);
for i = 1:numel(patches_to_analyze)
    my_scatter(patches_to_analyze_sorted(i),prt_sorted(i),plot_col(i,:),0.5);
end
ylabel('PRT (sec)');
xlabel('Patch Num');
axis square;

% plot PCs
for pc_num = 1:6
    pca_selected_patches = nan(numel(patches_to_analyze),N);
    for i = 1:numel(patches_to_analyze)
        pca_this = score_resid(patch_num==patches_to_analyze(i),pc_num);
        pca_selected_patches(i,1:numel(pca_this)) = pca_this;
    end

    % sort by prt
    [~,sort_idx] = sort(prt);
    pca_selected_patches_sort = pca_selected_patches(sort_idx,:);

    % plot
    subplot(2,7,pc_num+1); hold on;
    plot_col = winter(numel(patches_to_analyze));
    for i = 1:numel(patches_to_analyze)
        y_this = pca_selected_patches_sort(i,:);
        lh = plot(t,y_this);
        lh.Color = [plot_col(i,:) 0.8];
        max_idx = find(~isnan(y_this),1,'last');
        my_scatter(t(max_idx),y_this(max_idx),plot_col(i,:),0.5);
    end
    xlabel('Time on patch (sec)');
    ylabel('PC')
    title(sprintf('PC%d',pc_num));
    plot([min(prt) min(prt)],ylim,'k--');
    axis square;

    % plot significance
    for i = 1:size(pca_selected_patches,2)
        keep = ~isnan(pca_selected_patches(:,i));
        y_this = pca_selected_patches(keep,i);
        prt_this = prt(keep);
        if numel(y_this)>1
            [r,p] = corrcoef(y_this,prt_this);
            if p(1,2)<0.05
                plot(t(i),max(ylim)-0.5,'k.');
            end
        end
    end
    
    % plot rewards
    if strcmp(opt.patch_type,'R-nil')
        plot(0,max(ylim),'bv','MarkerFaceColor','b');
    elseif strcmp(opt.patch_type,'RR-nil')
        plot(0,max(ylim),'bv','MarkerFaceColor','b');
        plot(1,max(ylim),'bv','MarkerFaceColor','b');
    end
       
end

% SAME but colored by patch number (proxy for time in session)
% plot PRTs vs patch number in session
subplot(2,7,8); hold on;
plot_col = autumn(numel(patches_to_analyze));
for i = 1:numel(patches_to_analyze)
    my_scatter(patches_to_analyze(i),prt(i),plot_col(i,:),0.5);
end
ylabel('PRT (sec)');
xlabel('Patch Num');
axis square;

% plot PCs
for pc_num = 1:6
    pca_selected_patches = nan(numel(patches_to_analyze),N);
    for i = 1:numel(patches_to_analyze)
        pca_this = score_resid(patch_num==patches_to_analyze(i),pc_num);
        pca_selected_patches(i,1:numel(pca_this)) = pca_this;
    end

    % plot
    subplot(2,7,7+pc_num+1); hold on;
    plot_col = autumn(numel(patches_to_analyze));
    for i = 1:numel(patches_to_analyze)
        y_this = pca_selected_patches(i,:);
        lh = plot(t,y_this);
        lh.Color = [plot_col(i,:) 0.8];
        max_idx = find(~isnan(y_this),1,'last');
        my_scatter(t(max_idx),y_this(max_idx),plot_col(i,:),0.5);
    end
    xlabel('Time on patch (sec)');
    ylabel('PC')
    title(sprintf('PC%d',pc_num));
    plot([min(prt) min(prt)],ylim,'k--');
    axis square;

    % plot significance
    for i = 1:size(pca_selected_patches,2)
        keep = ~isnan(pca_selected_patches(:,i));
        y_this = pca_selected_patches(keep,i);
        x_this = patches_to_analyze(keep);
        if numel(y_this)>1
            [r,p] = corrcoef(y_this,x_this);
            if p(1,2)<0.05
                plot(t(i),max(ylim)-0.5,'k.');
            end
        end
    end
    
    % plot rewards
    if strcmp(opt.patch_type,'R-nil')
        plot(0,max(ylim),'bv','MarkerFaceColor','b');
    elseif strcmp(opt.patch_type,'RR-nil')
        plot(0,max(ylim),'bv','MarkerFaceColor','b');
        plot(1,max(ylim),'bv','MarkerFaceColor','b');
    end
       
end

save_figs(paths.figs,hfig,'png');

%% show time window of significance for all pc's stacked on each other

hfig = figure('Position',[300 300 700 300]); 
hfig.Name = sprintf('%s_PCs1-6 - significance overlay - %d uL patches - %s',opt.session,opt.rew_size,opt.patch_type);

subplot(1,2,1); hold on;
for pc_num = 1:6
    pca_selected_patches = nan(numel(patches_to_analyze),N);
    for i = 1:numel(patches_to_analyze)
        pca_this = score(patch_num==patches_to_analyze(i),pc_num);
        pca_selected_patches(i,1:numel(pca_this)) = pca_this;
    end

    % plot significance
    for i = 1:size(pca_selected_patches,2)
        keep = ~isnan(pca_selected_patches(:,i));
        y_this = pca_selected_patches(keep,i);
        x_this = prt(keep);
        if numel(y_this)>1
            [r,p] = corrcoef(y_this,x_this);
            if p(1,2)<0.05
                plot(t(i),pc_num,'k.');
            end
        end
    end
   
end
ylim([0.5 6.5]);
% plot rewards
if strcmp(opt.patch_type,'R-nil')
    plot(0,max(ylim),'bv','MarkerFaceColor','b');
elseif strcmp(opt.patch_type,'RR-nil')
    plot(0,max(ylim),'bv','MarkerFaceColor','b');
    plot(1,max(ylim),'bv','MarkerFaceColor','b');
end
xlabel('time on patch (sec)');
ylabel('PC');
title(sprintf('%s, %d uL patches:\noriginal PCs',opt.session,opt.rew_size),'Interpreter','none');

subplot(1,2,2); hold on;
for pc_num = 1:6
    pca_selected_patches = nan(numel(patches_to_analyze),N);
    for i = 1:numel(patches_to_analyze)
        pca_this = score_resid(patch_num==patches_to_analyze(i),pc_num);
        pca_selected_patches(i,1:numel(pca_this)) = pca_this;
    end

    % plot significance
    for i = 1:size(pca_selected_patches,2)
        keep = ~isnan(pca_selected_patches(:,i));
        y_this = pca_selected_patches(keep,i);
        x_this = prt(keep);
        if numel(y_this)>1
            [r,p] = corrcoef(y_this,x_this);
            if p(1,2)<0.05
                plot(t(i),pc_num,'k.');
            end
        end
    end
       
end
ylim([0.5 6.5]);
% plot rewards
if strcmp(opt.patch_type,'R-nil')
    plot(0,max(ylim),'bv','MarkerFaceColor','b');
elseif strcmp(opt.patch_type,'RR-nil')
    plot(0,max(ylim),'bv','MarkerFaceColor','b');
    plot(1,max(ylim),'bv','MarkerFaceColor','b');
end
xlabel('time on patch (sec)');
ylabel('PC');
title(sprintf('%s, %d uL patches:\nsession time regressed out',opt.session,opt.rew_size),'Interpreter','none');

save_figs(paths.figs,hfig,'png');