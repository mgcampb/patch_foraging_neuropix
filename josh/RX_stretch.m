%% Stretching it out
%  Can we learn anything about coding from stretching trials and averaging?
%  Take RXNil trials, stretch every trial to median of lengths, then average
%  -> Visualize peak-sorted PETHs and plot PC space

%% Set paths and basic analysis options

paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice';
paths.figs = '/Users/joshstern/Documents/UchidaLab_NeuralData/neural_data_figs'; % where to save figs

addpath(genpath('/Users/joshstern/Documents/UchidaLab_NeuralData'));

% analysis options
FR_calcOpt = struct;
FR_calcOpt.tbin = 0.02; % time bin for whole session rate matrix (in sec)
tbin_ms = FR_calcOpt.tbin * 1000;
FR_calcOpt.smoothSigma_time = 0.100; % gauss smoothing sigma for rate matrix (in sec)
FR_calcOpt.preLeave_buffer = 0;
FR_calcOpt.cortex_only = true;

sessions = dir(fullfile(paths.data,'*.mat'));
sessions = {sessions.name};

%% Extract FR matrices and timing information
FR_decVar = struct;
for sIdx = 1:25
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
for sIdx = 1:25
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

%% Iterate over trial types and make stretched averages

RXNil_peakSortPETH = cell(numel(sessions),6); 
for sIdx = 23:25
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    prts = data.patchCSL(:,3) - data.patchCSL(:,2);
    rew_barcode = rew_barcodes{sIdx};  
    nNeurons = size(FR_decVar(sIdx).fr_mat{1},1);  
    nTrials = length(prts);
    
    sec1ix = 1000 / tbin_ms;
    
    rew_counter = 1;
    for iRewsize = [1,2,4] 
        % Collect RXNil trials
        trialsR0Nil = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) < 0);
        trialsRRNil = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & rew_barcode(:,3) < 0);
        
        % Get median PRTs to stretch to 
        medianPRT_R0Nil_ix = round((1000 * median(prts(trialsR0Nil)) / tbin_ms)); 
        medianPostRewRT_RRNil_ix = round((1000 * (median(prts(trialsRRNil) - 1)) / tbin_ms)); 
        
        % collect stretched firing rates in cell 
        % note fancy stuff w/ RRNil; only stretch following rew reception
        R0Nil_tmpCell = cellfun(@(x) imresize(x,[nNeurons,medianPRT_R0Nil_ix]),FR_decVar(sIdx).fr_mat(trialsR0Nil),'UniformOutput',false);
        RRNil_tmpCell = cellfun(@(x) cat(2,x(:,1:sec1ix),imresize(x(:,sec1ix:end),[nNeurons,medianPostRewRT_RRNil_ix])) ... 
                                                            ,FR_decVar(sIdx).fr_mat(trialsRRNil),'UniformOutput',false);
        
        % get sorts from trials not in pool  
        dvar = "timesince"; 
        decVar_bins = linspace(0,2,41);
        peakSortOpt = struct; 
        peakSortOpt.trials = setdiff(1:nTrials,trialsR0Nil);  
        peakSortOpt.suppressVis = true; 
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
conditions = {"10Nil","20Nil","40Nil","11Nil","22Nil","44Nil"};
for sIdx = 23:25
    session = sessions{sIdx}; 
    session_title = session([1:2 end-6:end-4]);
    figure() 
    for cIdx = 1:6 
        subplot(2,3,cIdx);colormap('parula')
        imagesc(flipud(RXNil_peakSortPETH{sIdx,cIdx}))
        title(sprintf("%s %s",session_title,conditions{cIdx})) 
        caxis([-3,3])
    end
end 

%% Now perform PCA on concatenated averages 
RXNil_PCA = cell(numel(sessions),6);  
t_lens = cellfun(@(x) size(x,2),RXNil_peakSortPETH);
for sIdx = 23:25
    concat_RXNil = cat(2,RXNil_peakSortPETH{sIdx,:});
    [coeff,score,~,~,explained] = pca(concat_RXNil');
    start_ix = cumsum([1 t_lens(sIdx,1:end-1)]); 
    end_ix = cumsum(t_lens(sIdx,:));   
    for cIdx = 1:6
        RXNil_PCA{sIdx,cIdx} = score(start_ix(cIdx):end_ix(cIdx),1:10)';
    end
end 

%% Visualize PC trajectories 
colors = {[.5 1 1],[.75 .75 1],[1 .5 1],[0 1 1],[.5 .5 1],[1 0 1]}; 
for sIdx = 23:25
    figure(); hold on 
    for cIdx = 1:3
        plot(RXNil_PCA{sIdx,cIdx}(1,:),RXNil_PCA{sIdx,cIdx}(2,:),'linewidth',2,'color',colors{cIdx}) 
    end 
    
    % add arrows
    arrowSize = 5; 
    arrowGain = 0;
    arrowEdgeColor = 'k'; 
    for cIdx = 1:3
        % plot x's at second marks
        sec_ticks = 50:50:size(RXNil_PCA{sIdx,cIdx},2);  
        plot(RXNil_PCA{sIdx,cIdx}(1,sec_ticks),RXNil_PCA{sIdx,cIdx}(2,sec_ticks), 'kd', 'markerSize', 6, 'markerFaceColor',colors{cIdx});
        
        % plot first point as dot
        plot(RXNil_PCA{sIdx,cIdx}(1,1),RXNil_PCA{sIdx,cIdx}(2,1), 'ko', 'markerSize', 6, 'markerFaceColor',colors{cIdx});
        % last point as arrow
        penultimatePoint = [RXNil_PCA{sIdx,cIdx}(1,end-1), RXNil_PCA{sIdx,cIdx}(2,end-1)];
        lastPoint = [RXNil_PCA{sIdx,cIdx}(1,end), RXNil_PCA{sIdx,cIdx}(2,end)];
        vel = norm(lastPoint - penultimatePoint);
        xl = xlim();
        yl = ylim();
        axLim = [xl yl];
        aSize = arrowSize + arrowGain * vel;  % if asked (e.g. for movies) arrow size may grow with vel
        arrowMMC(penultimatePoint, lastPoint, [], aSize, axLim, colors{cIdx}, arrowEdgeColor);
    end
end
%% Old code %% 

% visualize distribution of PRTs over different trial types to see
% how kosher stretching is going to be
figure()
subplot(1,2,1)
histogram(prts(trialsR0Nil))
title(sprintf("%i0Nil Trial PRT Distribution",iRewsize))
subplot(1,2,2)
histogram(prts(trialsRRNil))
title(sprintf("%i%iNil Trial PRT Distribution",iRewsize,iRewsize))
