addpath(genpath('C:\code\patch_foraging_neuropix\malcolm\functions'));
addpath('C:\code\glmnet_matlab');

paths = struct;
paths.data = 'H:\My Drive\UchidaLab\PatchForaging\processed_neuropix_data';
paths.results_load = 'C:\data\patch_foraging_neuropix\GLM_output\run_20201114_all_sessions_model_comparison';
paths.result_save = 'C:\data\patch_foraging_neuropix\GLM_output\run_20201115_dropout';

opt = struct;
opt.brain_region = 'PFC';
opt.data_set = 'mb';
opt.pval_thresh = 0.05;

%% get sessions
session_all = dir(fullfile(paths.results_load,'*.mat'));
session_all = {session_all.name}';
for sesh_idx = 1:numel(session_all)
    session_all{sesh_idx} = session_all{sesh_idx}(1:end-4);
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
llh_contr = [];
sesh_all = [];
for sesh_idx = 1:numel(session_all)
    fprintf('Session %d/%d: %s\n',sesh_idx,numel(session_all),session_all{sesh_idx});
    
    fit = load(fullfile(paths.results_load,session_all{sesh_idx}));
    
    keep_cell = strcmp(fit.brain_region_rough,opt.brain_region);
    
    sig = fit.pval_full_vs_base(keep_cell)<opt.pval_thresh & sum(abs(fit.beta_all(fit.base_var==0,keep_cell))>0)'>0;

    sig_cells = fit.good_cells(keep_cell);
    sig_cells = sig_cells(sig);
    
    % make various dropout regressor matrices
    X_red = fit.X_full(:,fit.base_var(2:end)==1);
    X_drop_rew_kern = fit.X_full(:,~contains(fit.var_name(2:end),'RewKern'));
    X_drop_time_on_patch = fit.X_full(:,~contains(fit.var_name(2:end),'TimeOnPatch'));
    X_drop_total_rew = fit.X_full(:,~contains(fit.var_name(2:end),'TotalRew'));
    X_drop_time_since_rew = fit.X_full(:,~contains(fit.var_name(2:end),'TimeSinceRew'));
    
    % fit GLM for different models
    pb = ParforProgressbar(numel(sig_cells));
    opt_glmnet = glmnetSet;
    opt_glmnet.alpha = fit.opt.alpha; % alpha for elastic net
    log_llh_full = nan(numel(sig_cells),1);
    log_llh_red = nan(numel(sig_cells),1);
    log_llh_drop_rew_kern = nan(numel(sig_cells),1);
    log_llh_drop_time_on_patch = nan(numel(sig_cells),1);
    log_llh_drop_total_rew = nan(numel(sig_cells),1);
    log_llh_drop_time_since_rew = nan(numel(sig_cells),1);
    parfor cIdx = 1:numel(sig_cells)
        
        cell_idx = find(fit.good_cells==sig_cells(cIdx));
        
        y = fit.spikecounts(:,cell_idx);
        
        % full model
        X_this = fit.X_full;
        beta = fit.beta_all(:,cell_idx);
        r = exp([ones(size(X_this,1),1) X_this] * beta);
        log_llh_full(cIdx) = -nansum(r-y.*log(r)+log(factorial(y)))/sum(y);
        
        % reduced model
        X_this = X_red;
        glm_fit = cvglmnet(X_this,y,'poisson',opt_glmnet,[],[],fit.foldid);
        beta = cvglmnetCoef(glm_fit);
        r = exp([ones(size(X_this,1),1) X_this] * beta);
        log_llh_red(cIdx) = -nansum(r-y.*log(r)+log(factorial(y)))/sum(y);
        
        % dropout rew kern
        X_this = X_drop_rew_kern;
        glm_fit = cvglmnet(X_this,y,'poisson',opt_glmnet,[],[],fit.foldid);
        beta = cvglmnetCoef(glm_fit);
        r = exp([ones(size(X_this,1),1) X_this] * beta);
        log_llh_drop_rew_kern(cIdx) = -nansum(r-y.*log(r)+log(factorial(y)))/sum(y);
        
        % dropout time on patch
        X_this = X_drop_time_on_patch;
        glm_fit = cvglmnet(X_this,y,'poisson',opt_glmnet,[],[],fit.foldid);
        beta = cvglmnetCoef(glm_fit);
        r = exp([ones(size(X_this,1),1) X_this] * beta);
        log_llh_drop_time_on_patch(cIdx) = -nansum(r-y.*log(r)+log(factorial(y)))/sum(y);
        
        % dropout total rew
        X_this = X_drop_total_rew;
        glm_fit = cvglmnet(X_this,y,'poisson',opt_glmnet,[],[],fit.foldid);
        beta = cvglmnetCoef(glm_fit);
        r = exp([ones(size(X_this,1),1) X_this] * beta);
        log_llh_drop_total_rew(cIdx) = -nansum(r-y.*log(r)+log(factorial(y)))/sum(y);
        
        % dropout time since rew
        X_this = X_drop_time_since_rew;
        glm_fit = cvglmnet(X_this,y,'poisson',opt_glmnet,[],[],fit.foldid);
        beta = cvglmnetCoef(glm_fit);
        r = exp([ones(size(X_this,1),1) X_this] * beta);
        log_llh_drop_time_since_rew(cIdx) = -nansum(r-y.*log(r)+log(factorial(y)))/sum(y);
        
        pb.increment();
    end
    
    denom = log_llh_full-log_llh_red;
    numer = [log_llh_drop_rew_kern log_llh_drop_time_on_patch log_llh_drop_total_rew log_llh_drop_time_since_rew]-log_llh_red;
    llh_contr_this = 100*(1-numer./repmat(denom,1,4));
    llh_contr = [llh_contr; llh_contr_this];
    sesh_all = [sesh_all; sesh_idx*ones(numel(sig_cells),1)];

end