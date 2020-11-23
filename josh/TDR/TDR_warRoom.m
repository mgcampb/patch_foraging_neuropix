%% Targeted dimensionality reduction 
%  Driving questions: 
%  1. By looking at the low-dimensional components in neural
%     data that are correlated with task variables, can we predict leaving
%     with high fidelity?  
%  2. Does the trajectory of these low-dimensional components help us
%     understand a dynamic decision-making process?

%% Generic setup
paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice';
paths.figs = '/Users/joshstern/Documents/UchidaLab_NeuralData/neural_data_figs'; % where to save figs

addpath('/Users/joshstern/Documents/UchidaLab_NeuralData');

% analysis options
opt = struct;
opt.tbin = 0.02; % time bin for whole session rate matrix (in sec)
opt.smoothSigma_time = 0.1; % gauss smoothing sigma for rate matrix (in sec)
opt.patch_leave_buffer = .5; % in seconds; only takes within patch times up to this amount before patch leave
opt.min_fr = 0; % minimum firing rate (on patch, excluding buffer) to keep neurons 
opt.cortex_only = true;
tbin_ms = opt.tbin*1000;
sessions = dir(fullfile(paths.data,'*.mat'));
sessions = {sessions.name};

%% Load firing rate matrices, perform PCA
pca_trialed = cell(numel(sessions),1); 
mPFC_sessions = [1:8 10:13 15:18 23 25]; 
TDR_struct = struct; 
for i = 1:numel(mPFC_sessions)
    sIdx = mPFC_sessions(i);
    % Get the session name
    session = sessions{sIdx}(1:end-4); 
    dat = load(fullfile(paths.data,session));
    fprintf('Loading session %d/%d: %s...\n',sIdx,numel(sessions),session);
    
    % Get standardized PC transformation and smoothed fr mat
    opt.session = session; % session to analyze   
    new_load = true; % just for development purposes
    if new_load == true 
        [coeffs,fr_mat,good_cells,score,~,expl] = standard_pca_fn(paths,opt); 
    end
    % Get times to index firing rate matrix
    patchstop_ms = dat.patchCSL(:,2)*1000;
    patchleave_ms = dat.patchCSL(:,3)*1000;
    patchstop_ix = round(patchstop_ms / tbin_ms) + 1;
    patchleave_ix = round((patchleave_ms - 1000 * opt.patch_leave_buffer) / tbin_ms) + 1;
    
    % Gather firing rate matrices in trial form
    fr_mat_trials = cell(length(dat.patchCSL),1);
    for iTrial = 1:length(dat.patchCSL)
        fr_mat_trials{iTrial} = fr_mat(:,patchstop_ix(iTrial):patchleave_ix(iTrial)); 
    end  
    score_full = coeffs' * zscore(horzcat(fr_mat_trials{:}),[],2); % s.t. indexing will line up 
    
    % Get new indexing vectors for our just on patch matrix
    t_lens = cellfun(@(x) size(x,2),fr_mat_trials); 
    new_patchleave_ix = cumsum(t_lens);
    new_patchstop_ix = new_patchleave_ix - t_lens + 1;   
    
    % Similarly gather PCA projections to explain 75% variance in cell arra
    surpass_75 = find(cumsum(expl / sum(expl)) > .75,1);
    pca_trialed{sIdx} = cell(length(dat.patchCSL),1);
    for iTrial = 1:length(dat.patchCSL)
        pca_trialed{sIdx}{iTrial} = score_full(1:surpass_75,new_patchstop_ix(iTrial,1):new_patchleave_ix(iTrial,1)); 
    end 
    
    TDR_struct(sIdx).pca_trials = pca_trialed{sIdx}; 
end  

%% Generate "reward barcodes" to average firing rates  
rew_barcodes = cell(numel(sessions),1);
for i = 1:numel(mPFC_sessions) 
    sIdx = mPFC_sessions(i);
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    
    % Trial data
    patchstop_ms = data.patchCSL(:,2);
    patchleave_ms = data.patchCSL(:,3);
    rew_ms = data.rew_ts;
    patches = data.patches;
    patchCSL = data.patchCSL;
    prts = patchCSL(:,3) - patchCSL(:,2);
    floor_prts = floor(prts);
    rewsize = mod(patches(:,2),10);
    
    % make barcode matrices also want to know where we have no more rewards
    nTimesteps = 15;
    rew_barcode = zeros(length(patchCSL) , nTimesteps);
    for iTrial = 1:length(patchCSL)
        rew_indices = round(rew_ms(rew_ms >= patchstop_ms(iTrial) & rew_ms < patchleave_ms(iTrial)) - patchstop_ms(iTrial)) + 1;
        last_rew_ix = max(rew_indices);
        rew_barcode(iTrial , (last_rew_ix + 1):end) = -1; % set part of patch after last rew_ix = -1
        rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -2; % set part of patch after leave = -2
        rew_barcode(iTrial , rew_indices) = rewsize(iTrial);
    end
    rew_barcodes{sIdx} = rew_barcode;
end

%% Make task variables for regression
% - time on patch 
% - time since reward 
% - total rewards 
% - reward delivery event 
for i = 1:numel(mPFC_sessions)
    sIdx = mPFC_sessions(i);
    session = sessions{sIdx}(1:end-4); 
    dat = load(fullfile(paths.data,session)); 
    nTrials = length(dat.patchCSL); 
    rewsize = mod(dat.patches(:,2),10); 
    patchstop_sec = dat.patchCSL(:,2);
    patchleave_sec = dat.patchCSL(:,3);  
    prts = patchleave_sec - patchstop_sec; 
    floor_prts = floor(prts); 
    rew_sec = dat.rew_ts;  
    % index vectors
    patchstop_ms = dat.patchCSL(:,2)*1000;
    patchleave_ms = dat.patchCSL(:,3)*1000;
    patchstop_ix = round(patchstop_ms / tbin_ms) + 1;
    patchleave_ix = round((patchleave_ms - 1000 * opt.patch_leave_buffer) / tbin_ms) + 1; 
    prts_ix = patchleave_ix - patchstop_ix + 1;
    
    % make barcode matrices to make task variables
    nTimesteps = 15;
    rew_barcode = zeros(nTrials , nTimesteps); 
    rew_sec_cell = cell(nTrials,1);
    for iTrial = 1:nTrials
        rew_indices = round(rew_sec(rew_sec >= patchstop_sec(iTrial) & rew_sec < patchleave_sec(iTrial)) - patchstop_sec(iTrial)) + 1;
        last_rew_ix = max(rew_indices);
        rew_sec_cell{iTrial} = rew_indices(rew_indices > 1);
        rew_barcode(iTrial , (last_rew_ix + 1):end) = -1; % set part of patch after last rew_ix = -1
        rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -2; % set part of patch after leave = -2
        rew_barcode(iTrial , rew_indices) = rewsize(iTrial);
    end
    
    % Make decision variables
    time_on_patch = cell(nTrials,1);
    time_since_rew = cell(nTrials,1);  
    rew_num = cell(nTrials,1); 
    total_uL = cell(nTrials,1); 
    rew_binary_early = cell(nTrials,1);  
    rew_binary_late = cell(nTrials,1);  
    X_trials = cell(nTrials,1); 
    for iTrial = 1:nTrials
        trial_len_ix = prts_ix(iTrial);
        time_on_patch{iTrial} = (1:trial_len_ix) * tbin_ms / 1000;
        time_since_rew{iTrial} = (1:trial_len_ix) * tbin_ms / 1000;  
        rew_num{iTrial} = zeros(trial_len_ix,1);  
        total_uL{iTrial} = zeros(trial_len_ix,1);   
        rew_binary_early{iTrial} = zeros(trial_len_ix,1); 
        rew_binary_late{iTrial} = zeros(trial_len_ix,1); 
        
        for r = 1:numel(rew_sec_cell{iTrial})
            rew_ix = (rew_sec_cell{iTrial}(r) - 1) * 1000 / tbin_ms;
            time_since_rew{iTrial}(rew_ix:end) =  (1:length(time_since_rew{iTrial}(rew_ix:end))) * tbin_ms / 1000;
            rew_num{iTrial}(rew_ix:end) = r;  
            total_uL{iTrial}(rew_ix:end) = r * rewsize(iTrial); 
            rew_binary_early{iTrial}(rew_ix:min(trial_len_ix,rew_ix + round(500/tbin_ms))) = rewsize(iTrial);
            rew_binary_late{iTrial}(min(trial_len_ix,rew_ix + round(500/tbin_ms)):min(trial_len_ix,rew_ix + round(1000/tbin_ms))) = rewsize(iTrial);
        end 
        X_trials{iTrial} = [time_on_patch{iTrial}' time_since_rew{iTrial}' rew_num{iTrial} total_uL{iTrial} rew_binary_early{iTrial} rew_binary_late{iTrial}]';
    end  
    
    TDR_struct(sIdx).X_trials = X_trials;
end  


%% Perform Regression 
%  would be nice to do this cross-validated for visualization
for i = 1:numel(mPFC_sessions)
    sIdx = mPFC_sessions(i);   
    X_full = cat(2,TDR_struct(sIdx).X_trials{:});
    pca_full = cat(2,TDR_struct(sIdx).pca_trials{:});

    beta = mvregress(pca_full',X_full');
    
    TDR_struct(sIdx).beta = beta; 
end  

%% visualize correlation between axes  
close all
regressors = ["Time on Patch","Time Since Reward","Reward Number","Total uL","0:500 msec Since Rew","500:1000 msec Since Rew"]; 
RdBu = flipud(cbrewer('div','RdBu',100)); 
for i = 18
    sIdx = mPFC_sessions(i);   
    session_title = ['m' sessions{sIdx}(1:2) ' ' sessions{sIdx}(end-6) '/' sessions{sIdx}(end-5:end-4)]; 
    X_full = cat(2,TDR_struct(sIdx).X_trials{:});
    figure('Renderer', 'painters', 'Position', [300 300 1200 400])
    subplot(1,3,1)
    imagesc(corrcoef(TDR_struct(sIdx).beta)) 
    xticklabels(regressors) 
    xtickangle(45) 
    yticklabels(regressors) 
    ytickangle(45)   
    colormap(RdBu)  
    caxis([-1,1])
    colorbar() 
    title(sprintf("%s Correlations between Regression Axes",session_title))  
    subplot(1,3,2)
    imagesc(corrcoef(X_full')) 
    xticklabels(regressors) 
    xtickangle(45) 
    yticklabels(regressors) 
    ytickangle(45)   
    colormap(RdBu)  
    caxis([-1,1])
    colorbar() 
    title(sprintf("%s Correlations between Regressors",session_title))  
    subplot(1,3,3)
    imagesc(corrcoef(TDR_struct(sIdx).beta) - corrcoef(X_full')) 
    xticklabels(regressors) 
    xtickangle(45) 
    yticklabels(regressors) 
    ytickangle(45)   
    colormap(RdBu)  
    caxis([-1,1])
    colorbar() 
    title(sprintf("%s Corrcoef(Regressor Axes) - Corrcoef(Regressor) Correlations",session_title)) 
    
    TDR_struct(sIdx).betaCorrcoef = corrcoef(TDR_struct(sIdx).beta);
end

%% Project PCs onto taskVar axes 
RX_means = cell(numel(mPFC_sessions),6); 
RXX_means = cell(numel(mPFC_sessions),12);
for i = 1:numel(mPFC_sessions)
    sIdx = mPFC_sessions(i);   
    session = sessions{sIdx}(1:end-4); 
    dat = load(fullfile(paths.data,session)); 
    patchstop_ms = dat.patchCSL(:,2);
    patchleave_ms = dat.patchCSL(:,3);  
    prts = patchleave_ms - patchstop_ms;  
    nTrials = length(prts);  
    rew_barcode = rew_barcodes{sIdx};
    
    TDR_struct(sIdx).projected_trials = cell(nTrials,1); 
    
    for iTrial = 1:numel(TDR_struct(sIdx).X_trials) 
        TDR_struct(sIdx).projected_trials{iTrial} = TDR_struct(sIdx).beta' * TDR_struct(sIdx).pca_trials{iTrial};
    end 
    
    % Now average projections over RX and RXX trials for visualization 
    sec2ix = round(2000 / tbin_ms); 
    sec3ix = round(3000 / tbin_ms);  
    rewsizes = [1,2,4];
    for r = 1:numel(rewsizes) 
        iRewsize = rewsizes(r);
        
        % RX trialtypes
        trialsR0x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) <= 0 & prts > 2.55);
        trialsRRx = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & prts > 2.55);  
        % RXX trialtypes
        trialsR00x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) <= 0 & rew_barcode(:,3) <= 0 & prts > 3.55);
        trialsRR0x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & rew_barcode(:,3) <= 0 & prts > 3.55);
        trialsR0Rx = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == 0 & rew_barcode(:,3) == iRewsize & prts > 3.55);
        trialsRRRx = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & rew_barcode(:,3) == iRewsize & prts > 3.55);
    
        % average RX
        trialsR0x_cell = cellfun(@(x) x(:,1:sec2ix),TDR_struct(sIdx).projected_trials(trialsR0x),'UniformOutput',false);
        trialsRRx_cell = cellfun(@(x) x(:,1:sec2ix),TDR_struct(sIdx).projected_trials(trialsRRx),'UniformOutput',false);
        RX_means{sIdx,r} = mean(cat(3,trialsR0x_cell{:}),3); 
        RX_means{sIdx,r + 3} = mean(cat(3,trialsRRx_cell{:}),3); 
        
        % average RX
        trialsR00x_cell = cellfun(@(x) x(:,1:sec3ix),TDR_struct(sIdx).projected_trials(trialsR00x),'UniformOutput',false);
        trialsRR0x_cell = cellfun(@(x) x(:,1:sec3ix),TDR_struct(sIdx).projected_trials(trialsRR0x),'UniformOutput',false);
        trialsR0Rx_cell = cellfun(@(x) x(:,1:sec3ix),TDR_struct(sIdx).projected_trials(trialsR0Rx),'UniformOutput',false);
        trialsRRRx_cell = cellfun(@(x) x(:,1:sec3ix),TDR_struct(sIdx).projected_trials(trialsRRRx),'UniformOutput',false); 
        RXX_means{sIdx,(r - 1) * 4 + 1} = mean(cat(3,trialsR00x_cell{:}),3); 
        RXX_means{sIdx,(r - 1) * 4 + 2} = mean(cat(3,trialsRR0x_cell{:}),3); 
        RXX_means{sIdx,(r - 1) * 4 + 3} = mean(cat(3,trialsR0Rx_cell{:}),3);  
        RXX_means{sIdx,(r - 1) * 4 + 4} = mean(cat(3,trialsRRRx_cell{:}),3); 
    end
end

%% Visualize projected axes RX trials  
colors = {[.5 1 1],[.75 .75 1],[1 .5 1],[0 1 1],[.5 .5 1],[1 0 1]};  
conds = 1:6;  
rIdx = [1,3];
close all
for i = 1:numel(mPFC_sessions)
    sIdx = mPFC_sessions(i);   
    session_title = ['m' sessions{sIdx}(1:2) ' ' sessions{sIdx}(end-6) '/' sessions{sIdx}(end-5:end-4)]; 
    figure();hold on
    for condIdx = conds 
        if ~isempty(RX_means{sIdx,condIdx})
            plot(RX_means{sIdx,condIdx}(rIdx(1),:),RX_means{sIdx,condIdx}(rIdx(2),:),'color',colors{condIdx},'linewidth',1.5);hold on
            sec_ticks = 50;   
            plot(RX_means{sIdx,condIdx}(rIdx(1),1),RX_means{sIdx,condIdx}(rIdx(2),1), 'ko', 'markerSize', 6, 'markerFaceColor',colors{condIdx});
            plot(RX_means{sIdx,condIdx}(rIdx(1),sec_ticks),RX_means{sIdx,condIdx}(rIdx(2),sec_ticks), 'kd', 'markerSize', 6, 'markerFaceColor',colors{condIdx}); 
        end
    end  
    
    % get axis limits to draw arrows
    xl = xlim();
    yl = ylim();
    arrowSize = 5; 
    arrowGain = 0;
    arrowEdgeColor = 'k';
    
    for condIdx = conds  
        if ~isempty(RX_means{sIdx,condIdx})
            % for arrow, figure out last two points, and (if asked) supress the arrow if velocity is below a threshold.
            penultimatePoint = [RX_means{sIdx,condIdx}(rIdx(1),end-1), RX_means{sIdx,condIdx}(rIdx(2),end-1)];
            lastPoint = [RX_means{sIdx,condIdx}(rIdx(1),end), RX_means{sIdx,condIdx}(rIdx(2),end)];
            vel = norm(lastPoint - penultimatePoint);

            axLim = [xl yl];
            aSize = arrowSize + arrowGain * vel;  % if asked (e.g. for movies) arrow size may grow with vel
            arrowMMC(penultimatePoint, lastPoint, [], aSize, axLim, colors{condIdx}, arrowEdgeColor); 
        end
    end

    xlabel(sprintf("Projection onto %s Axis",regressors(rIdx(1))))
    ylabel(sprintf("Projection onto %s Axis",regressors(rIdx(2)))) 
    title(session_title)
end