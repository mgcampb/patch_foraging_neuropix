%% Visualization to prepare for dynamical systems analysis 
%  We are making a 3 x 2 plot showing 10, 20, 40 trials
%  Row 1: Binscatter to show density of vistation in PC space 
%  Row 2: Heatmap of PC1/3 gradient magnitude + vector field  

%% Generic setup
paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice';
paths.figs = '/Users/joshstern/Documents/UchidaLab_NeuralData/neural_data_figs'; % where to save figs

addpath(genpath('/Users/joshstern/Documents/UchidaLab_NeuralData'));

% analysis options
opt = struct;
opt.tbin = 0.02; % time bin for whole session rate matrix (in sec)
opt.smoothSigma_time = 0.1; % gauss smoothing sigma for rate matrix (in sec)
opt.patch_leave_buffer = 0; % in seconds; only takes within patch times up to this amount before patch leave
opt.min_fr = 0; % minimum firing rate (on patch, excluding buffer) to keep neurons 
opt.cortex_only = true;
tbin_ms = opt.tbin*1000;
sessions = dir(fullfile(paths.data,'*.mat'));
sessions = {sessions.name};

%% Load firing rate matrices, perform PCA
pca_trialed = cell(numel(sessions),1); 
mPFC_sessions = [1:8 10:13 15:18 23 25];
for i = 5
    sIdx = mPFC_sessions(i);
    % Get the session name
    session = sessions{sIdx}(1:end-4); 
    dat = load(fullfile(paths.data,session));
    fprintf('Loading session %d/%d: %s...\n',sIdx,numel(sessions),session);
    
    % Get standardized PC transformation and smoothed fr mat
    opt.session = session; % session to analyze   
    new_load = true; % just for development purposes
    if new_load == true 
        [coeffs,fr_mat,good_cells,score] = standard_pca_fn(paths,opt); 
    end
    % Get times to index firing rate matrix
    patchstop_ms = dat.patchCSL(:,2)*1000;
    patchleave_sec = dat.patchCSL(:,3)*1000;
    patchstop_ix = round(patchstop_ms / tbin_ms) + 1;
    patchleave_ix = min(round((patchleave_sec - 1000 * opt.patch_leave_buffer) / tbin_ms) + 1,size(fr_mat,2));
    
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
    
    % Similarly gather PCA 1:20 projections in trial cell array 
    pca_trialed{sIdx} = cell(length(dat.patchCSL),1);
    for iTrial = 1:length(dat.patchCSL)
        pca_trialed{sIdx}{iTrial} = score_full(1:20,new_patchstop_ix(iTrial,1):new_patchleave_ix(iTrial,1)); 
    end 
end 

%% Group RX PC traces in RX_data struct 

RX_data = cell(numel(sessions),1);
for i = 1:numel(mPFC_sessions) 
    sIdx = mPFC_sessions(i);
    RX_data{sIdx} = cell(6,2);
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    session_title = session([1:2 end-2:end]);
    patches = data.patches;
    patchCSL = data.patchCSL;
    
    % reinitialize ms vectors
    patchstop_sec = patchCSL(:,2);
    patchleave_sec = patchCSL(:,3);
    rew_sec = data.rew_ts;
    sec1ix = 1000/tbin_ms;
    sec2ix = 2000/tbin_ms;
    
    % Trial level features
    prts = patchCSL(:,3) - patchCSL(:,2);
    floor_prts = floor(prts);
    rewsize = mod(patches(:,2),10);
    
    % make barcode matrices
    nTimesteps = 10;
    rew_barcode = zeros(length(patchCSL) , nTimesteps); 
    rew_indices = cell(length(patchCSL),1);
    for iTrial = 1:length(patchCSL)
        rew_indices{iTrial} = round(rew_sec(rew_sec >= patchstop_sec(iTrial) & rew_sec < patchleave_sec(iTrial)) - patchstop_sec(iTrial)) + 1;
        rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -1; % set part of patch after leave = -1
        rew_barcode(iTrial , rew_indices{iTrial}) = rewsize(iTrial); 
        rew_indices{iTrial} = rew_indices{iTrial} - 1;
    end
  
    % Collect PC traces from RX trial types
    rew_counter = 1;
    for iRewsize = [1,2,4] 
        iRewsize_trials = find(rewsize == iRewsize);
        trialsR0x_2sec = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == 0 & prts > 2.55);
        trialsRRx_2sec = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & prts > 2.55); 
        
        % Add pc data up to first reward reception (or leave)
        tmp_morePC_cell = cell(numel(iRewsize_trials),1);
        for j = 1:numel(iRewsize_trials)  
            iTrial = iRewsize_trials(j); 
            t_len = size(pca_trialed{sIdx}{iTrial},2);
            if ~isempty(rew_indices{iTrial}(rew_indices{iTrial} > 0)) % cut off at first rew
                end_ix = min(t_len,min(rew_indices{iTrial}(rew_indices{iTrial} > 0)) / tbin_ms * 1000); 
            else 
                end_ix = t_len; 
            end 
            tmp_morePC_cell{j} = pca_trialed{sIdx}{iTrial}(:,1:end_ix);
        end
        
%         tmp_allPC_cell = pca_trialed{sIdx}(trialsR0x_more); 
        RX_data{sIdx}{rew_counter,1} = cat(2,tmp_morePC_cell);
        tmp_2secPC_cell = cellfun(@(x) x(:,1:sec2ix),pca_trialed{sIdx}(trialsR0x_2sec),'UniformOutput',false);
        RX_data{sIdx}{rew_counter,2} = mean(cat(3,tmp_2secPC_cell{:}),3);
        
        % same for RR
        tmp_morePC_cell = cell(numel(trialsRRx_2sec),1);
        for j = 1:numel(trialsRRx_2sec)
            iTrial = trialsRRx_2sec(j); 
            t_len = size(pca_trialed{sIdx}{iTrial},2); 
            if ~isempty(rew_indices{iTrial}(rew_indices{iTrial} > 1))
                end_ix = min(t_len,min(rew_indices{iTrial}(rew_indices{iTrial} > 1)) / tbin_ms * 1000); 
            else 
                end_ix = t_len; 
            end
            tmp_morePC_cell{j} = pca_trialed{sIdx}{iTrial}(:,1:end_ix);
        end
        %         tmp_allPC_cell = pca_trialed{sIdx}(trialsR0x_more);
        RX_data{sIdx}{rew_counter+3,1} = cat(2,tmp_morePC_cell);
        tmp_2secPC_cell = cellfun(@(x) x(:,1:sec2ix),pca_trialed{sIdx}(trialsRRx_2sec),'UniformOutput',false); 
        RX_data{sIdx}{rew_counter + 3,2} = mean(cat(3,tmp_2secPC_cell{:}),3); 

        rew_counter = rew_counter + 1;
    end
end 

%% Visualize mean trajectories   
close all
colors = {[.5 1 1],[.75 .75 1],[1 .5 1],[0 1 1],[.5 .5 1],[1 0 1]};  
conds = 1:6; 
plot_pcs = [1,2];

for i = 18
    sIdx = mPFC_sessions(i);
    session = sessions{sIdx}(1:end-4);
    session_title = session([1:2 end-2:end]); 
    arrowSize = 5; 
    arrowGain = 0;
    arrowEdgeColor = 'k'; 
    figure();hold on
    % plot mean condition trajectories
    for condIdx = conds 
        pc1 = RX_data{sIdx}{condIdx,2}(plot_pcs(1),:); 
        pc2 = RX_data{sIdx}{condIdx,2}(plot_pcs(2),:); 
        plot(pc1,pc2,'color',colors{condIdx},'linewidth',1.5)
        plot(pc1(1),pc2(1), 'ko', 'markerSize', 6, 'markerFaceColor',colors{condIdx});
    end  
    xl = xlim; 
    yl = ylim;
    
    % Now add arrows for extra credit 
    for condIdx = conds
        pc1 = RX_data{sIdx}{condIdx,2}(plot_pcs(1),:); 
        pc2 = RX_data{sIdx}{condIdx,2}(plot_pcs(2),:); 
        penultimatePoint = [pc1(end-1), pc2(end-1)];
        lastPoint = [pc1(end),pc2(end)];
        vel = norm(lastPoint - penultimatePoint);
        axLim = [xl yl];
        aSize = arrowSize + arrowGain * vel;  % if asked (e.g. for movies) arrow size may grow with vel
        arrowMMC(penultimatePoint, lastPoint, [], aSize, axLim, colors{condIdx}, arrowEdgeColor);
    end 
    xlim([xl(1) - 2, xl(2) + 2]) 
    ylim([yl(1) - 2, yl(2) + 2]) 
    title(sprintf("%s RX PC Traces",session_title)) 
    xlabel(sprintf("PC %i",plot_pcs(1)))
    ylabel(sprintf("PC %i",plot_pcs(2)))
end

%% Now make six-pane plot  
close all
plot_pcs = [1,2]; 
Nbins = 25; 
conds = 1:3; 
condNames = {"10","20","40","11","22","44"};
for i = 18
    sIdx = mPFC_sessions(i);
    session = sessions{sIdx}(1:end-4);
    session_title = session([1:2 end-2:end]); 
    
    figure(1)  
    %  Row 1: Binscatter to show density of vistation in PC space
    for cIdx = 1:numel(conds)
        subplot(1,3,cIdx) 
        cond_pca_full = cat(2,RX_data{sIdx}{conds(cIdx),1}{:}); 
        h = binscatter(cond_pca_full(plot_pcs(1),:),cond_pca_full(plot_pcs(2),:),Nbins);     
%         xt_og = xticks();
%         yt_og = yticks(); 
%         disp(xt_og)
        values = flipud(h.Values'); 
        imagesc(values) 
        colormap([0 0 0; parula])
        colorbar(gca,'off')   
        xlabel(sprintf("PC%i",plot_pcs(1))) 
        ylabel(sprintf("PC%i",plot_pcs(2)))  
%         xticklabels(xt_og) 
%         yticklabels(yt_og) 
        xticks([]) 
        yticks([]) 
        title(condNames{cIdx}) 
    end  
    suptitle(sprintf("%s State space occupency frequency",session_title))
    
    figure(2)
    %  Row 2: Heatmap of PC1/3 gradient magnitude 
    % !!!!!!!!! NEED TO MAKE BIN EDGES THE SAME OR ADD PADDING... BUT THE
    % AXES ARE NOT THE SAME CURRENTLY !!!!!!!!!
    for cIdx = 1:numel(conds)
        subplot(2,3,cIdx);
        cond_pca_full = cat(2,RX_data{sIdx}{conds(cIdx),1}{:});
        dState = cond_pca_full(:,2:end) - cond_pca_full(:,1:end-1);
        norm_dState = vecnorm(dState)';
        cond_pca_full = cond_pca_full(:,1:end-1); % make dimensions match
        PC1 = discretize(cond_pca_full(plot_pcs(1),:)',Nbins);  % programatically assign here?
        PC2 = discretize(cond_pca_full(plot_pcs(2),:)',Nbins);
        pca_table = table(PC1, PC2,norm_dState);
        h = heatmap(pca_table,"PC1",...
            "PC2",...
            'ColorVariable',"norm_dState");
        imagesc(flipud(h.ColorData));colormap([0 0 0; parula])
        if cIdx == 1
            cl = [max(0,mean(norm_dState) - std(norm_dState)),...
                mean(norm_dState) + std(norm_dState)];
        end
        caxis(cl)
    end 
    
    % Row 3: Vector field  
    for cIdx = 1:numel(conds)  
        subplot(2,3,cIdx)
        cond_pca_full = cat(2,RX_data{sIdx}{conds(cIdx),1}{:});  
        dState = cond_pca_full(:,2:end) - cond_pca_full(:,1:end-1);  
        norm_dState = vecnorm(dState)';
        cond_pca_full = cond_pca_full(:,1:end-1); % make dimensions match    
        % discretize state space and gradients
        [PC1,edgePC1] = discretize(cond_pca_full(plot_pcs(1),:)',Nbins);  % programatically assign here?
        [PC2,edgePC2] = discretize(cond_pca_full(plot_pcs(2),:)',Nbins);  
        edge_bin1 = edgePC1(2) - edgePC1(1);
        edge_bin2 = edgePC2(2) - edgePC2(1);
        centBinPC1 = edgePC1(1:end-1) + edge_bin1 / 2;
        centBinPC2 = edgePC2(1:end-1) + edge_bin2 / 2;
        %         dPC1 = discretize(dState(plot_pcs(1),:)',edgePC1); % pc1 gradient
        %         dPC2 = discretize(dState(plot_pcs(2),:)',edgePC2); % pc2 gradient

        dPC1 = dState(plot_pcs(1),:)';
        dPC2 = dState(plot_pcs(2),:)';
        pca_gradient_table = table(PC1,PC2,dPC1,dPC2); 
        xlabel(sprintf("PC%i",plot_pcs(1))) 
        ylabel(sprintf("PC%i",plot_pcs(2)))  
        xticks([])
        yticks([]) 
        title(condNames{cIdx})  
        title(sprintf("%s |F'(x)|",condNames{cIdx}))
        % get average gradients per point in binned PC space 
        figure(3)
        dPC1_hmap = heatmap(pca_gradient_table,"PC1",...
            "PC2",...
            'ColorVariable',"dPC1"); 
        dPC1_hmap = dPC1_hmap.ColorData; 
        dPC2_hmap = heatmap(pca_gradient_table,"PC1",...
                                               "PC2",...
                                               'ColorVariable',"dPC2"); 
        dPC2_hmap = dPC2_hmap.ColorData;    
        close()
        figure(2) 
        subplot(2,3,cIdx+3)
        % get rid of outliers 
        muDPC1 = nanmean(dPC1_hmap(:));
        sigmaDPC1 = nanstd(dPC1_hmap(:)); 
        muDPC2 = nanmean(dPC2_hmap(:)); 
        sigmaDPC2 = nanstd(dPC2_hmap(:));   
        dPC1_hmap(dPC1_hmap > muDPC1 + 1.96 * sigmaDPC1) = muDPC1 + 1.96 * sigmaDPC1;
        dPC1_hmap(dPC1_hmap < muDPC1 - 1.96 * sigmaDPC1) = muDPC1 - 1.96 * sigmaDPC1;
        dPC2_hmap(dPC2_hmap > muDPC2 + 1.96 * sigmaDPC2) = muDPC2 + 1.96 * sigmaDPC2;
        dPC2_hmap(dPC2_hmap < muDPC2 - 1.96 * sigmaDPC2) = muDPC2 - 1.96 * sigmaDPC2; 
        normDState = sqrt(dPC1_hmap.^2 + dPC2_hmap.^2); 
        [x,y] = meshgrid(centBinPC1,centBinPC2);
        quiverC2D(x,y,dPC1_hmap,dPC2_hmap,5)
        xlim([xl(1) - 5, xl(2) + 5]) 
        ylim([yl(1) - 5, yl(2) + 5]) 
        xlabel(sprintf("PC%i",plot_pcs(1))) 
        ylabel(sprintf("PC%i",plot_pcs(2)))   
        title(sprintf("%s Flow Field",condNames{cIdx}))
    end 
    suptitle(sprintf("%s Gradient Visualization",session_title))
    
end



