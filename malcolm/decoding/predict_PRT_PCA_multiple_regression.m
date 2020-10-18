%% script to plot neural activity (PC1 and example neuron)
% for large reward patches with just one reward at t=0
% to see if there's a correlation with leave times
% MGC 6/23/2020

% ** TO DO: add firing rate cutoff ** 
% ** TO DO: add cross-validation **
session = {'78_20200310','78_20200311'};
rew_size = [1 2 4];

for i1 = 1:numel(session)
for i2 = 1:numel(rew_size)
    
paths = struct;
paths.data = 'C:\Users\malcg\Dropbox (Personal)\UchidaLab\processed_neuropix_data';
paths.figs = 'C:\figs\patch_foraging_neuropix\decoding\pca'; % where to save figs

addpath(genpath('C:\code\HGRK_analysis_tools'));
addpath(genpath('C:\code\patch_foraging_neuropix\malcolm\functions'));

% analysis options
opt = struct;
opt.session = session{i1}; % session to analyze
opt.tbin = 0.02; % time bin for whole session rate matrix (in sec)
opt.smoothSigma_time = 0.1; % gauss smoothing sigma for rate matrix (in sec)

% for concatenating patches
opt.patch_type = 'R-nil'; % options: 'R-nil' or 'RR-nil'
opt.patch_leave_buffer = 0.5; % in seconds; only takes within patch times up to this amount before patch leave
opt.rew_size = rew_size(i2);

% num pcs to use for regression
opt.num_pcs = 6;

% regression on shuffled data
opt.num_shuf = 1000;

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

%% Multiple linear regression with cross-validation
% use data with session time and session time^2 regressed out

T = ceil(max(prt));
t = 0:opt.tbin:T;
N = numel(t);



% plot PCs
pca_selected_patches = nan(numel(patches_to_analyze),N,opt.num_pcs);
for i = 1:numel(patches_to_analyze)
    pca_this = score_resid(patch_num==patches_to_analyze(i),1:opt.num_pcs);
    pca_selected_patches(i,1:size(pca_this,1),:) = pca_this;
end

% only keep up to first leave time
T = floor(min(prt)/opt.tbin)*opt.tbin;
t = 0:opt.tbin:T;
N = numel(t);
pca_selected_patches = pca_selected_patches(:,1:N,:);
pca_selected_patches = permute(pca_selected_patches,[1 3 2]);

% multiple linear regression
R2 = nan(size(pca_selected_patches,3),1);
for i = 1:size(pca_selected_patches,3)
    X = [ones(numel(prt),1) pca_selected_patches(:,:,i)];
    y = prt;
    beta = X\y;
    y_pred = X*beta;
    y_resid = y-y_pred;
    R2(i) = 1-sum(y_resid.^2)/sum((y-mean(y)).^2);
end

%% SHUFFLE
R2_shuf = nan(size(pca_selected_patches,3),opt.num_shuf);
for i = 1:size(pca_selected_patches,3)
    X = [ones(numel(prt),1) pca_selected_patches(:,:,i)];
    for shuf_idx = 1:opt.num_shuf
        y = prt(randsample(numel(prt),numel(prt)));
        beta = X\y;
        y_pred = X*beta;
        y_resid = y-y_pred;
        R2_shuf(i,shuf_idx) = 1-sum(y_resid.^2)/sum((y-mean(y)).^2);
    end
end

% shuffle p-value
shuf_pval = size(pca_selected_patches,3);
for i = 1:size(pca_selected_patches,3)
    shuf_pval(i) = sum(R2_shuf(i,:)>=R2(i))/opt.num_shuf;
end
    


%% PLOT

hfig = figure('Position',[100 300 700 400]); hold on;
hfig.Name = sprintf('%s_PCs1-6 - %d uL patches - %s - multiple regression',opt.session,opt.rew_size,opt.patch_type);

% plot R2
h1 = plot(t,R2);
ylim([0 1]);
ylabel('R squared');
xlabel('time on patch (sec)');
title(sprintf('%s PCs1-%d\n%duL %s',opt.session,opt.num_pcs,opt.rew_size,opt.patch_type),'Interpreter','none');

% plot rewards
if strcmp(opt.patch_type,'R-nil')
    plot(0,max(ylim),'bv','MarkerFaceColor','b','HandleVisibility','off');
elseif strcmp(opt.patch_type,'RR-nil')
    plot(0,max(ylim),'bv','MarkerFaceColor','b','HandleVisibility','off');
    plot(1,max(ylim),'bv','MarkerFaceColor','b','HandleVisibility','off');
end
plot([min(prt) min(prt)],[min(ylim) max(ylim)],'k--','HandleVisibility','off');
text(min(prt),max(ylim),'first patch leave','HorizontalAlignment','right');

% plot shuffle distributions
shadedErrorBar(t,mean(R2_shuf,2),std(R2_shuf,[],2));
legend('data','shuffle (permute PRT)','Location','northeastoutside');

% plot significance
for i = 1:numel(shuf_pval)
    if shuf_pval(i)<0.05
        plot(t(i),0.95,'k.','HandleVisibility','off');
    end
end

save_figs(paths.figs,hfig,'png');

end
end
