%% Stretching it out
%  Can we learn anything about coding from stretching trials and averaging?
%  Take RXNil trials, stretch every trial to median of lengths, then average
%  -> Visualize peak-sorted PETHs and plot PC space

%% Set paths and basic analysis options

paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice';
% paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mgc';
paths.figs = '/Users/joshstern/Documents/UchidaLab_NeuralData/neural_data_figs'; % where to save figs 
paths.rampIDs = 'Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/ramping_neurons';
paths.transients_table = '/Users/joshstern/Documents/UchidaLab_NeuralData/patch_foraging_neuropix/josh/structs/transients_table.mat';
load(paths.transients_table); 
paths.sig_cells = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/glm_results/sig_cells/sig_cells_mb_cohort_PFC.mat';
load(paths.sig_cells);  
addpath('/Users/joshstern/Documents/UchidaLab_NeuralData');

% analysis options
FR_calcOpt = struct;
FR_calcOpt.tbin = 0.02; % time bin for whole session rate matrix (in sec)
tbin_ms = FR_calcOpt.tbin * 1000;
FR_calcOpt.smoothSigma_time = 0.100; % gauss smoothing sigma for rate matrix (in sec)
FR_calcOpt.preLeave_buffer = 0;
FR_calcOpt.region_selection = "PFC";  
mPFC_sessions = [1:8 10:13 14:18 23 25]; 
mgcPFC_sessions = [1:2 5 7]; 
mgcSTR_sessions = [3:4 6 8:9];
sessions = dir(fullfile(paths.data,'*.mat'));
sessions = {sessions.name}; 

%% Extract FR matrices and timing information
FR_decVar = struct;
for i = 1:numel(mPFC_sessions)
    sIdx = mPFC_sessions(i);  
%     if ismember(sIdx,mgcPFC_sessions) 
%         FR_calcOpt.region_selection = "PFC";  
%         FR_calcOpt.cortex_only = false;
%     elseif ismember(sIdx,mgcSTR_sessions)  
%         FR_calcOpt.region_selection = "STR";   
%         FR_calcOpt.cortex_only = false;
%     else 
%         disp("Warning: no region for this session")
%     end
    
    FR_decVar_tmp = genSeqStructs(paths,sessions,FR_calcOpt,sIdx);
    % assign to sIdx
    FR_decVar(sIdx).fr_mat = FR_decVar_tmp.fr_mat;
    FR_decVar(sIdx).goodcell_IDs = FR_decVar_tmp.goodcell_IDs;
    FR_decVar(sIdx).decVarTime = FR_decVar_tmp.decVarTime;
    FR_decVar(sIdx).decVarTimeSinceRew = FR_decVar_tmp.decVarTimeSinceRew;
    FR_decVar(sIdx).cell_depths = FR_decVar_tmp.spike_depths;
end

%% Generate "reward barcodes" to average firing rates
rew_barcodes = cell(numel(sessions),1);
prts = cell(numel(sessions),1); 
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
    i_prts = patchCSL(:,3) - patchCSL(:,2);
    floor_prts = floor(i_prts);
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
    prts{sIdx} = i_prts; 
end

%% Iterate over trial types and make stretched averages

RXNil_peakSortPETH = cell(numel(mPFC_sessions),6); 
for i = 1:numel(mPFC_sessions)
    sIdx = mPFC_sessions(i); 
    session = sessions{sIdx}(1:end-4); 
    session_name = session([1:2 end-2:end]); 
    data = load(fullfile(paths.data,session));
    i_prts = prts{sIdx};
    rew_barcode = rew_barcodes{sIdx};  
    
    % use results from driscoll selection
    session_table = transients_table(transients_table.Session == session_name & transients_table.Region == "PFC",:);
    peak_rew1plus = session_table.Rew1plus_peak_pos;
    glm_clust = session_table.GLM_Cluster;
    nTrials = length(i_prts);
    neuron_selection = ~isnan(peak_rew1plus) & peak_rew1plus < 1.25 & peak_rew1plus > .25;
    nNeurons = length(find(neuron_selection));  
    sec1ix = 1000 / tbin_ms;
    
    rew_counter = 1;
    for iRewsize = [1,2,4] 
        % Collect RXNil trials
        trialsR0Nil = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) < 0 & i_prts > 1.55);
        trialsRRNil = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & rew_barcode(:,3) < 0 & i_prts > 1.55);
        
        % Get median PRTs - 1 to stretch to 
        medianPRT_R0Nil_ix = round((1000 * median(i_prts(trialsR0Nil) - 1) / tbin_ms)); 
        medianPostRewRT_RRNil_ix = round((1000 * (median(i_prts(trialsRRNil) - 1)) / tbin_ms)); 
        
        % collect stretched firing rates in cell 
        % note fancy stuff: only stretch following last rew reception 
        % also do this for R0Nil to match part of trial that is stretched
%         R0Nil_tmpCell = cellfun(@(x) imresize(x,[nNeurons,medianPRT_R0Nil_ix]),FR_decVar(sIdx).fr_mat(trialsR0Nil),'UniformOutput',false);
        R0Nil_tmpCell = cellfun(@(x) cat(2,x(neuron_selection,1:sec1ix),imresize(x(neuron_selection,sec1ix:end),[nNeurons,medianPRT_R0Nil_ix])) ... 
                                                            ,FR_decVar(sIdx).fr_mat(trialsR0Nil),'UniformOutput',false);
        RRNil_tmpCell = cellfun(@(x) cat(2,x(neuron_selection,1:sec1ix),imresize(x(neuron_selection,sec1ix:end),[nNeurons,medianPostRewRT_RRNil_ix])) ... 
                                                            ,FR_decVar(sIdx).fr_mat(trialsRRNil),'UniformOutput',false);
        
        % get sorts from trials not in pool  
        dvar = "timesince"; 
        decVar_bins = linspace(0,2,41);
        peakSortOpt = struct; 
        peakSortOpt.trials = setdiff(1:nTrials,trialsR0Nil); % peak sort by heldout trials
        peakSortOpt.suppressVis = true; 
        peakSortOpt.neurons = find(neuron_selection); 
        [~,neuron_orderR0Nil] = peakSortPETH(FR_decVar(sIdx),dvar,decVar_bins,peakSortOpt); 
        peakSortOpt.trials = setdiff(1:nTrials,trialsRRNil);   
        [~,neuron_orderRRNil] = peakSortPETH(FR_decVar(sIdx),dvar,decVar_bins,peakSortOpt); 
        
        % take average over stretched trials
        if ~isempty(trialsR0Nil)
            RXNil_peakSortPETH{sIdx,rew_counter} = zscore(mean(cat(3,R0Nil_tmpCell{:}),3),[],2);
            RXNil_peakSortPETH{sIdx,rew_counter} = RXNil_peakSortPETH{sIdx,rew_counter}(neuron_orderR0Nil,:);
        else
            RXNil_peakSortPETH{sIdx,rew_counter} = zeros(10,10);
        end
        
        if ~isempty(trialsRRNil)
            RXNil_peakSortPETH{sIdx,rew_counter+3} = zscore(mean(cat(3,RRNil_tmpCell{:}),3),[],2);
            RXNil_peakSortPETH{sIdx,rew_counter+3} = RXNil_peakSortPETH{sIdx,rew_counter+3}(neuron_orderRRNil,:);
        else
            RXNil_peakSortPETH{sIdx,rew_counter+3} = zeros(10,10);
        end
        
        rew_counter = rew_counter + 1;
    end
end 

%% Now visualize   
close all
conditions = ["10Nil","20Nil","40Nil","11Nil","22Nil","44Nil"]; 
% session_names = ["PFC mc2 10/05" "PFC mc2 10/16" "STR mc2 10/20" "STR mc4 10/24" "PFC mc4 10/25" "STR mc4 10/26" "PFC mc4 10/27" "STR mc4 10/31" "STR mc4 11/01"];

for i = 1:numel(mPFC_sessions)
    sIdx = mPFC_sessions(i); 
    session = sessions{sIdx}; 
    session_title = session([1:2 end-6:end-4]); %  session_names(sIdx); %  session([1:2 end-6:end-4]);
    figure() 
    for cIdx = 1:6 
        subplot(2,3,cIdx);colormap('parula')
        imagesc(flipud(RXNil_peakSortPETH{sIdx,cIdx}))
        title(sprintf("%s %s",session_title,conditions{cIdx})) 
        caxis([-3,3])  
        t_len = size(RXNil_peakSortPETH{sIdx,cIdx},2);
        xticks(0:50:t_len) 
        xticklabels((0:50:t_len) * tbin_ms / 1000)
    end
end 

%% Now separate RNil trials by early vs late PRT 
% start w/ just m80? 
RXNil_prtPETH = cell(numel(sessions),6);
for i = 1:numel(mPFC_sessions) 
    sIdx = mPFC_sessions(i); 
    session = sessions{sIdx}(1:end-4); 
    session_name = session([1:2 end-2:end]); 
    data = load(fullfile(paths.data,session));
    i_prts = prts{sIdx};
    rew_barcode = rew_barcodes{sIdx};  
    
    % use results from driscoll selection
    session_table = transients_table(transients_table.Session == session_name & transients_table.Region == "PFC",:);
    peak_rew1plus = session_table.Rew1plus_peak_pos;
    glm_clust = session_table.GLM_Cluster;
    neuron_selection = ~isnan(peak_rew1plus);%  & peak_rew1plus < 1.25 & peak_rew1plus > .25;
    nNeurons = length(find(neuron_selection));  
%     nNeurons = length(find(~isnan(peak_rew1plus)));  
    nTrials = length(i_prts);
    
    vis_ix = 2000 / tbin_ms; % how much of trial to visualizes
    
    rew_counter = 1;
    for iRewsize = [1,2,4]
        % Collect RXNil trials
        trialsR0Nil = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) < 0 & i_prts > 2.55);
        
        if length(trialsR0Nil) >= 3
        [~,~,prt_bin] = histcounts(i_prts(trialsR0Nil),[0 quantile(i_prts(trialsR0Nil),2) max(i_prts(trialsR0Nil))]);
        early_rnil_trials = trialsR0Nil(prt_bin == 1); 
        late_rnil_trials = trialsR0Nil(prt_bin == 3); 
        
        % collect stretched firing rates in cell 
        R0Nil_tmpEarly = FR_decVar(sIdx).fr_mat(early_rnil_trials); 
        R0Nil_tmpEarly = cellfun(@(x) x(neuron_selection,1:vis_ix),R0Nil_tmpEarly,'un',0);
        R0Nil_tmpLate = FR_decVar(sIdx).fr_mat(late_rnil_trials); 
        R0Nil_tmpLate = cellfun(@(x) x(neuron_selection,1:vis_ix),R0Nil_tmpLate,'un',0);
        
        % get sorts from trials not in pool  
        dvar = "timesince"; 
        decVar_bins = linspace(0,2,41);
        peakSortOpt = struct; 
        peakSortOpt.trials = setdiff(1:nTrials,trialsR0Nil); % peak sort by heldout trials
        peakSortOpt.suppressVis = true; 
        peakSortOpt.neurons = find(neuron_selection); 
        [~,neuron_orderR0Nil] = peakSortPETH(FR_decVar(sIdx),dvar,decVar_bins,peakSortOpt); 
        
        % take avg
%         RXNil_prtPETH{sIdx,rew_counter} = zscore(mean(cat(3,R0Nil_tmpEarly{:}),3),[],2);
        RXNil_prtPETH{sIdx,rew_counter} = mean(cat(3,R0Nil_tmpEarly{:}),3);
        RXNil_prtPETH{sIdx,rew_counter} = RXNil_prtPETH{sIdx,rew_counter}(neuron_orderR0Nil,:); % sort
%         RXNil_prtPETH{sIdx,rew_counter+3} = zscore(mean(cat(3,R0Nil_tmpLate{:}),3),[],2);
        RXNil_prtPETH{sIdx,rew_counter+3} = mean(cat(3,R0Nil_tmpLate{:}),3);
        RXNil_prtPETH{sIdx,rew_counter+3} = RXNil_prtPETH{sIdx,rew_counter+3}(neuron_orderR0Nil,:); % sort
        
        else
            RXNil_prtPETH{sIdx,rew_counter} = zeros(10,10);
            RXNil_prtPETH{sIdx,rew_counter+3} = zeros(10,10);
        
        end
        rew_counter = rew_counter + 1;
    end
end

%% Now visualize   
close all
conditions = ["1Nil Early Leave","2Nil Early Leave","4Nil Early Leave","1Nil Late Leave","2Nil Late Leave","4Nil Late Leave"]; 
cool3 = [0 1 1 ; .5 .5 1; 1 0 1];   % 1 .5 1
cool3_light = [.5 1 1;.75 .75 1;1 .5 1]; 
colors = [cool3_light ; cool3]; 
% {[.5 1 1],[.75 .75 1],[1 .5 1],[0 1 1],[.5 .5 1],[1 0 1]};
for i = 19 % 1:numel(mPFC_sessions)
    sIdx = mPFC_sessions(i); 
    session = sessions{sIdx}; 
    session_title = session([1:2 end-6:end-4]); %  session_names(sIdx); %  session([1:2 end-6:end-4]);
    slopes = nan(6,1); 
    slopes_SE = nan(6,1); 
    hfig = figure('Renderer', 'painters', 'Position', [10 10 900 600]);
    hfig.Name = [session_title '_seqSlope']; 
    for cIdx = 1:6 
        if cIdx < 4
            hfig(cIdx) = subplot(2,4,cIdx);
        else
            hfig(cIdx) = subplot(2,4,cIdx+1); 
        end
        colormap('parula')
        imagesc(flipud(zscore(RXNil_prtPETH{sIdx,cIdx},[],2)));hold on
%         imagesc(flipud(RXNil_prtPETH{sIdx,cIdx}./max(RXNil_prtPETH{sIdx,cIdx},[],2)))
        title(sprintf("%s %s",session_title,conditions{cIdx})) 
        colorbar()
        caxis([-3,3])  
        t_len = size(RXNil_peakSortPETH{sIdx,cIdx},2);
        xticks(0:50:t_len) 
        xticklabels((0:50:t_len) * tbin_ms / 1000)
        
        if cIdx >= 4
            xlabel("Time (sec)")
        end
        
        this_peth = zscore(RXNil_prtPETH{sIdx,cIdx},[],2);
%         this_peth = RXNil_prtPETH{sIdx,cIdx};
        if ~isequal(zeros(10,10),this_peth) 
            [neurons,times] = find(this_peth > 0);
            activity = this_peth(this_peth > 0);
            
            % weighted linear regression on first second sequence
            mdl = fitlm(times,neurons,'Intercept',false,'Weights',activity);
            slopes(cIdx) = mdl.Coefficients.Estimate / (tbin_ms / 1000); % normalized progression speed
            slopes_SE(cIdx) = mdl.Coefficients.SE / (tbin_ms / 1000); 
            plot(1:vis_ix,size(RXNil_prtPETH{sIdx,cIdx},1) - predict(mdl,(1:vis_ix)'),'color',colors(cIdx,:),'linewidth',2)
        end
    end 
    hfig(7) = subplot(2,4,4);
    b = bar([slopes(1:3) slopes(4:6)],'FaceColor','Flat');
    b(1).CData = cool3_light; b(2).CData = cool3; 
    x = [b(1).XEndPoints b(2).XEndPoints];
    hold on; 
    errorbar(x,slopes,slopes_SE,'k.') 
    xticks([1 2 3]) 
    xticklabels(["1 uL","2 uL","4 uL"])  
    ylabel("Cells/sec") 
    title(sprintf("Fit Activity Progression Slope"))
    ylim([50,70]) 
    pos3 = get(hfig(3),'Position');
    pos6 = get(hfig(6),'Position');
    pos7 = get(hfig(7),'Position');
    set(hfig(7),'Position',[pos7(1) new pos7(3:end)])
%     save_figs(paths.figs,hfig(cIdx),'png'); 
%     close(hfig);
end 

%% fit slopes  
for i = 19 % 1:numel(mPFC_sessions)   
    sIdx = mPFC_sessions(i); 
    figure();hold on
    for cIdx = 1:6 
        if cIdx < 4
            subplot(1,3,cIdx);hold on 
        else 
            subplot(1,3,cIdx-3);hold on 
        end
        this_peth = zscore(RXNil_prtPETH{sIdx,cIdx},[],2);
%         this_peth = RXNil_prtPETH{sIdx,cIdx};
        if ~isequal(zeros(10,10),this_peth)
            [neurons,times] = find(this_peth > 0);
            activity = this_peth(this_peth > 0);
            
            % weighted linear regression on first second sequence
            mdl = fitlm(times,neurons,'Intercept',false,'Weights',activity);
            prog_slopes(iTrial) = mdl.Coefficients.Estimate(1) / size(this_peth,1); % normalized progression speed
%             plot(predict(mdl,(1:vis_ix)'),'linewidth',2);
            [ypred,yci] = predict(mdl,(1:vis_ix)'); 
            shadedErrorBar((1:vis_ix),ypred,yci)
        end
    end
end

%% Old code 

% %% Now perform PCA on concatenated averages 
% RXNil_PCA = cell(numel(sessions),6);  
% t_lens = cellfun(@(x) size(x,2),RXNil_peakSortPETH);
% for sIdx = 1:numel(sessions)
%     concat_RXNil = cat(2,RXNil_peakSortPETH{sIdx,:});
%     [coeff,score,~,~,explained] = pca(concat_RXNil');
%     start_ix = cumsum([1 t_lens(sIdx,1:end-1)]); 
%     end_ix = cumsum(t_lens(sIdx,:));   
%     for cIdx = 1:6
%         RXNil_PCA{sIdx,cIdx} = score(start_ix(cIdx):end_ix(cIdx),1:10)';
%     end
% end 
% 
% %% Visualize PC trajectories 
% colors = {[.5 1 1],[.75 .75 1],[1 .5 1],[0 1 1],[.5 .5 1],[1 0 1]}; 
% for sIdx = 1:numel(sessions)
%     figure(); hold on 
%     for cIdx = 4:6
%         plot(RXNil_PCA{sIdx,cIdx}(1,:),RXNil_PCA{sIdx,cIdx}(2,:),'linewidth',2,'color',colors{cIdx}) 
%     end 
%     
%     % add arrows
%     arrowSize = 5; 
%     arrowGain = 0;
%     arrowEdgeColor = 'k'; 
%     for cIdx = 4:6
%         % plot x's at second marks
%         sec_ticks = 50:50:size(RXNil_PCA{sIdx,cIdx},2);  
%         plot(RXNil_PCA{sIdx,cIdx}(1,sec_ticks),RXNil_PCA{sIdx,cIdx}(2,sec_ticks), 'kd', 'markerSize', 6, 'markerFaceColor',colors{cIdx});
%         
%         % plot first point as dot
%         plot(RXNil_PCA{sIdx,cIdx}(1,1),RXNil_PCA{sIdx,cIdx}(2,1), 'ko', 'markerSize', 6, 'markerFaceColor',colors{cIdx});
%         % last point as arrow
%         penultimatePoint = [RXNil_PCA{sIdx,cIdx}(1,end-1), RXNil_PCA{sIdx,cIdx}(2,end-1)];
%         lastPoint = [RXNil_PCA{sIdx,cIdx}(1,end), RXNil_PCA{sIdx,cIdx}(2,end)];
%         vel = norm(lastPoint - penultimatePoint);
%         xl = xlim();
%         yl = ylim();
%         axLim = [xl yl];
%         aSize = arrowSize + arrowGain * vel;  % if asked (e.g. for movies) arrow size may grow with vel
%         arrowMMC(penultimatePoint, lastPoint, [], aSize, axLim, colors{cIdx}, arrowEdgeColor);
%     end
% end
% %% Old code %% 
% 
% % visualize distribution of PRTs over different trial types to see
% % how kosher stretching is going to be
% figure()
% subplot(1,2,1)
% histogram(prts(trialsR0Nil))
% title(sprintf("%i0Nil Trial PRT Distribution",iRewsize))
% subplot(1,2,2)
% histogram(prts(trialsRRNil))
% title(sprintf("%i%iNil Trial PRT Distribution",iRewsize,iRewsize))
