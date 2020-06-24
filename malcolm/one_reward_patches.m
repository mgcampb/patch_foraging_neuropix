%% script to plot neural activity (PC1 and example neuron)
% for large reward patches with just one reward at t=0
% to see if there's a correlation with leave times
% MGC 6/23/2020

paths = struct;
paths.data = 'C:\Users\malcg\Dropbox (Personal)\UchidaLab\processed_neuropix_data';
paths.figs = 'C:\figs\patch_foraging_neuropix\one_reward_patches'; % where to save figs

addpath(genpath('C:\code\HGRK_analysis_tools'));
addpath(genpath('C:\code\patch_foraging_neuropix\malcolm\functions'));

% analysis options
opt = struct;
opt.session = '80_20200315'; % session to analyze
% opt.cellid = 368; % example cell to plot
opt.tbin = 0.02; % time bin for whole session rate matrix (in sec)
opt.smoothSigma_time = 0.1; % gauss smoothing sigma for rate matrix (in sec)

%% load data
dat = load(fullfile(paths.data,opt.session));

%% PCA on fat firing rate matrix: neurons (N) by time (T) where T is the length of the session

good_cells = dat.sp.cids(dat.sp.cgs==2);

% time bins
opt.tstart = 0;
opt.tend = max(dat.sp.st);
tbinedge = opt.tstart:opt.tbin:opt.tend;
tbincent = tbinedge(1:end-1)+opt.tbin/2;

% compute firing rate mat
fr_mat = calcFRVsTime(good_cells,dat,opt);

% only keep "in-patch" times
in_patch = false(size(tbincent));
for i = 1:size(dat.patchCSL,1)
    in_patch(tbincent>=dat.patchCSL(i,2) & tbincent<=dat.patchCSL(i,3)) = true;
end
fr_mat_in_patch = fr_mat(:,in_patch);
tbincent = tbincent(in_patch);

% take zscore
fr_mat_zscore = my_zscore(fr_mat_in_patch);

% pca on firing rate matrix
[coeffs,score,~,~,expl] = pca(fr_mat_zscore');

%% look at patches with only one reward (the first one)

% get patch num for each patch
patch_num = nan(size(tbincent));
for i = 1:size(dat.patchCSL,1)
    patch_num(tbincent>=dat.patchCSL(i,2) & tbincent<=dat.patchCSL(i,3)) = i;
end

one_rew_patches = find(dat.patches(:,3)==1 & mod(dat.patches(:,2),10)==4);
prt = dat.patches(one_rew_patches,5);

%% regress out time in session and time^2 to remove slowly varying changes

X = [ones(size(tbincent)); tbincent; tbincent.^2]'; % include time
beta = X\score;
score_resid = score-X*beta;
beta2 = X\score_resid;

hfig = figure('Position',[200 200 600 1000]);
hfig.Name = sprintf('one_rew_patches_%s_PCs1-6 original and residual after removing time and time squared',opt.session);
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
hfig.Name = sprintf('one_rew_patches_%s_PCs1-6 - 4 uL patches - only reward at t=0',opt.session);

% plot PRTs vs patch number in session
subplot(2,7,1); hold on;
plot_col = cool(numel(one_rew_patches));
[prt_sorted,sort_idx] = sort(prt);
one_rew_patches_sorted = one_rew_patches(sort_idx);
for i = 1:numel(one_rew_patches)
    my_scatter(one_rew_patches_sorted(i),prt_sorted(i),plot_col(i,:),0.5);
end
ylabel('PRT (sec)');
xlabel('Patch Num');
axis square;

% plot PCs
for pc_num = 1:6
    pca_one_rew_patches = nan(numel(one_rew_patches),N);
    for i = 1:numel(one_rew_patches)
        pca_this = score(patch_num==one_rew_patches(i),pc_num);
        pca_one_rew_patches(i,1:numel(pca_this)) = pca_this;
    end

    % sort by prt
    [~,sort_idx] = sort(prt);
    pca_one_rew_patches_sort = pca_one_rew_patches(sort_idx,:);

    % plot
    subplot(2,7,pc_num+1); hold on;
    plot_col = cool(numel(one_rew_patches));
    for i = 1:numel(one_rew_patches)
        y_this = pca_one_rew_patches_sort(i,:);
        lh = plot(t,y_this);
        lh.Color = [plot_col(i,:) 0.8];
        max_idx = find(~isnan(y_this),1,'last');
        my_scatter(t(max_idx),y_this(max_idx),plot_col(i,:),0.5);
    end
    xlabel('Time on patch (sec)');
    ylabel('PC')
    % title(sprintf('%s\n4 uL patches, only reward at t=0',opt.session),'Interpreter','none')
    title(sprintf('PC%d',pc_num));
    plot([min(prt) min(prt)],ylim,'k--');
    axis square;

    % plot significance
    for i = 1:size(pca_one_rew_patches,2)
        keep = ~isnan(pca_one_rew_patches(:,i));
        y_this = pca_one_rew_patches(keep,i);
        prt_this = prt(keep);
        if numel(y_this)>1
            [r,p] = corrcoef(y_this,prt_this);
            if p(1,2)<0.05
                plot(t(i),max(ylim)-0.5,'k.');
            end
        end
    end
       
end

% SAME but colored by patch number (proxy for time in session)
% plot PRTs vs patch number in session
subplot(2,7,8); hold on;
plot_col = autumn(numel(one_rew_patches));
for i = 1:numel(one_rew_patches)
    my_scatter(one_rew_patches(i),prt(i),plot_col(i,:),0.5);
end
ylabel('PRT (sec)');
xlabel('Patch Num');
axis square;

% plot PCs
for pc_num = 1:6
    pca_one_rew_patches = nan(numel(one_rew_patches),N);
    for i = 1:numel(one_rew_patches)
        pca_this = score(patch_num==one_rew_patches(i),pc_num);
        pca_one_rew_patches(i,1:numel(pca_this)) = pca_this;
    end

    % plot
    subplot(2,7,7+pc_num+1); hold on;
    plot_col = autumn(numel(one_rew_patches));
    for i = 1:numel(one_rew_patches)
        y_this = pca_one_rew_patches(i,:);
        lh = plot(t,y_this);
        lh.Color = [plot_col(i,:) 0.8];
        max_idx = find(~isnan(y_this),1,'last');
        my_scatter(t(max_idx),y_this(max_idx),plot_col(i,:),0.5);
    end
    xlabel('Time on patch (sec)');
    ylabel('PC')
    % title(sprintf('%s\n4 uL patches, only reward at t=0',opt.session),'Interpreter','none')
    title(sprintf('PC%d',pc_num));
    plot([min(prt) min(prt)],ylim,'k--');
    axis square;

    % plot significance
    for i = 1:size(pca_one_rew_patches,2)
        keep = ~isnan(pca_one_rew_patches(:,i));
        y_this = pca_one_rew_patches(keep,i);
        x_this = one_rew_patches(keep);
        if numel(y_this)>1
            [r,p] = corrcoef(y_this,x_this);
            if p(1,2)<0.05
                plot(t(i),max(ylim)-0.5,'k.');
            end
        end
    end
       
end

save_figs(paths.figs,hfig,'png');

%% plot pc traces for these patches: TIME AND TIME^2 REGRESSED OUT

T = ceil(max(prt));
t = 0:opt.tbin:T;
N = numel(t);

hfig = figure('Position',[100 300 2100 800]); hold on;
hfig.Name = sprintf('one_rew_patches_%s_PCs1-6 - 4 uL patches - only reward at t=0 - regressed out time in session and time squared',opt.session);

% plot PRTs vs patch number in session
subplot(2,7,1); hold on;
plot_col = cool(numel(one_rew_patches));
[prt_sorted,sort_idx] = sort(prt);
one_rew_patches_sorted = one_rew_patches(sort_idx);
for i = 1:numel(one_rew_patches)
    my_scatter(one_rew_patches_sorted(i),prt_sorted(i),plot_col(i,:),0.5);
end
ylabel('PRT (sec)');
xlabel('Patch Num');
axis square;

% plot PCs
for pc_num = 1:6
    pca_one_rew_patches = nan(numel(one_rew_patches),N);
    for i = 1:numel(one_rew_patches)
        pca_this = score_resid(patch_num==one_rew_patches(i),pc_num);
        pca_one_rew_patches(i,1:numel(pca_this)) = pca_this;
    end

    % sort by prt
    [~,sort_idx] = sort(prt);
    pca_one_rew_patches_sort = pca_one_rew_patches(sort_idx,:);

    % plot
    subplot(2,7,pc_num+1); hold on;
    plot_col = cool(numel(one_rew_patches));
    for i = 1:numel(one_rew_patches)
        y_this = pca_one_rew_patches_sort(i,:);
        lh = plot(t,y_this);
        lh.Color = [plot_col(i,:) 0.8];
        max_idx = find(~isnan(y_this),1,'last');
        my_scatter(t(max_idx),y_this(max_idx),plot_col(i,:),0.5);
    end
    xlabel('Time on patch (sec)');
    ylabel('PC')
    % title(sprintf('%s\n4 uL patches, only reward at t=0',opt.session),'Interpreter','none')
    title(sprintf('PC%d',pc_num));
    plot([min(prt) min(prt)],ylim,'k--');
    axis square;

    % plot significance
    for i = 1:size(pca_one_rew_patches,2)
        keep = ~isnan(pca_one_rew_patches(:,i));
        y_this = pca_one_rew_patches(keep,i);
        prt_this = prt(keep);
        if numel(y_this)>1
            [r,p] = corrcoef(y_this,prt_this);
            if p(1,2)<0.05
                plot(t(i),max(ylim)-0.5,'k.');
            end
        end
    end
       
end

% SAME but colored by patch number (proxy for time in session)
% plot PRTs vs patch number in session
subplot(2,7,8); hold on;
plot_col = autumn(numel(one_rew_patches));
for i = 1:numel(one_rew_patches)
    my_scatter(one_rew_patches(i),prt(i),plot_col(i,:),0.5);
end
ylabel('PRT (sec)');
xlabel('Patch Num');
axis square;

% plot PCs
for pc_num = 1:6
    pca_one_rew_patches = nan(numel(one_rew_patches),N);
    for i = 1:numel(one_rew_patches)
        pca_this = score_resid(patch_num==one_rew_patches(i),pc_num);
        pca_one_rew_patches(i,1:numel(pca_this)) = pca_this;
    end

    % plot
    subplot(2,7,7+pc_num+1); hold on;
    plot_col = autumn(numel(one_rew_patches));
    for i = 1:numel(one_rew_patches)
        y_this = pca_one_rew_patches(i,:);
        lh = plot(t,y_this);
        lh.Color = [plot_col(i,:) 0.8];
        max_idx = find(~isnan(y_this),1,'last');
        my_scatter(t(max_idx),y_this(max_idx),plot_col(i,:),0.5);
    end
    xlabel('Time on patch (sec)');
    ylabel('PC')
    % title(sprintf('%s\n4 uL patches, only reward at t=0',opt.session),'Interpreter','none')
    title(sprintf('PC%d',pc_num));
    plot([min(prt) min(prt)],ylim,'k--');
    axis square;

    % plot significance
    for i = 1:size(pca_one_rew_patches,2)
        keep = ~isnan(pca_one_rew_patches(:,i));
        y_this = pca_one_rew_patches(keep,i);
        x_this = one_rew_patches(keep);
        if numel(y_this)>1
            [r,p] = corrcoef(y_this,x_this);
            if p(1,2)<0.05
                plot(t(i),max(ylim)-0.5,'k.');
            end
        end
    end
       
end

save_figs(paths.figs,hfig,'png');

%% plot example cell for these patches

% % get firing rate trace for this cell for these patches
% T = ceil(max(prt));
% t = 0:opt.tbin:T;
% N = numel(t);
% fr_one_rew_patches = nan(numel(one_rew_patches),N);
% for i = 1:numel(one_rew_patches)
%     fr_this = fr_mat_in_patch(good_cells==opt.cellid,patch_num==one_rew_patches(i));
%     fr_one_rew_patches(i,1:numel(fr_this)) = fr_this;
% end
% 
% % sort by prt
% [~,sort_idx] = sort(prt);
% fr_one_rew_patches_sort = fr_one_rew_patches(sort_idx,:);
% 
% % plot
% hfig = figure('Position',[300 300 600 500]); hold on;
% hfig.Name = sprintf('one_rew_patches_%s_c%d',opt.session,opt.cellid);
% plot_col = cool(numel(one_rew_patches));
% for i = 1:numel(one_rew_patches)
%     y_this = fr_one_rew_patches_sort(i,:);
%     lh = plot(t,y_this);
%     lh.Color = [plot_col(i,:) 0.8];
%     max_idx = find(~isnan(y_this),1,'last');
%     my_scatter(t(max_idx),y_this(max_idx),plot_col(i,:),0.5);
% end
% xlabel('Time on patch (sec)');
% ylabel('Firing rate (Hz)')
% title(sprintf('%s\nCELL %d, 4 uL patches, only reward at t=0',opt.session,opt.cellid),'Interpreter','none')
% plot([min(prt) min(prt)],ylim,'k--');
% 
% % plot significance
% for i = 1:size(fr_one_rew_patches,2)
%     keep = ~isnan(fr_one_rew_patches(:,i));
%     y_this = fr_one_rew_patches(keep,i);
%     prt_this = prt(keep);
%     if numel(y_this)>1
%         [r,p] = corrcoef(y_this,prt_this);
%         if p(1,2)<0.05
%             my_scatter(t(i),max(ylim)-2,'k',1);
%         end
%     end
% end
% 
% save_figs(paths.figs,hfig,'png');
