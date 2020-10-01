%% jPCA on neural data between cue and stop 
%  also a good exercise in getting basic timewarping approaches going 

paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/80';
paths.figs = '/Users/joshstern/Documents/UchidaLab_NeuralData/neural_data_figs'; % where to save figs

addpath(genpath('/Users/joshstern/Documents/UchidaLab_NeuralData/HGK_analysis_tools'));
addpath(genpath('/Users/joshstern/Documents/UchidaLab_NeuralData'));

% analysis options
frCalc_opt = struct;
frCalc_opt.tbin = 0.02; % time bin for whole session rate matrix (in sec) 
tbin_ms = frCalc_opt.tbin * 1000;
frCalc_opt.smoothSigma_time = 0.050; % gauss smoothing sigma for rate matrix (in sec)

sessions = dir(fullfile(paths.data,'*.mat'));
sessions = {sessions.name};

%% Extract FR matrix betw cue, stop
cueStop_frCell = {numel(sessions)};

for sIdx = 3:3
    % for name
    session = sessions{sIdx}(1:end-4);
    
    % load data
    dat = load(fullfile(paths.data,session)); 
    patches = dat.patches;
    patchCSL = dat.patchCSL;
    fprintf('Loading session %d/%d: %s...\n',sIdx,numel(sessions),session);
    good_cells = dat.sp.cids(dat.sp.cgs==2); 
    
    % time bins
    frCalc_opt.tstart = 0;
    frCalc_opt.tend = max(dat.sp.st);
    
    % behavioral events to align to 
    patchcue_ms = patchCSL(:,1)*1000;
    patchstop_ms = patchCSL(:,2)*1000;  
    patchleave_ms = patchCSL(:,3)*1000;   
    prts = patchleave_ms - patchstop_ms; 
    cs_ms = patchstop_ms - patchcue_ms;

    new_fr_mat = true;
    if new_fr_mat == true
        % compute firing rate matrix
        tic
        [fr_mat, ~] = calcFRVsTime(good_cells,dat,frCalc_opt); % calc from full matrix
        toc
    end 
    
    preCue = 0;
    % create index vectors from our update timestamp vectors 
    patchcue_ix = round((patchcue_ms-preCue) / tbin_ms) + 1;
    patchstop_ix = round(patchstop_ms / tbin_ms) + 1;
    
    % make struct
    cueStop_frCell{sIdx} = {length(patchCSL)};
    for iTrial = 1:length(patchCSL)
        cueStop_frCell{sIdx}{iTrial} = fr_mat(:,patchcue_ix(iTrial):patchstop_ix(iTrial));
    end 
    
    figure() 
    scatter(cs_ms,prts) 
    title("Cue-stop time vs PRT") 
    xlabel("Cue-stop duration") 
    ylabel("PRT")
    
end 

%% Do Cue-stop vs PRT comparison only for 4... trials  

% reinitialize ms vectors
patchstop_ms = dat.patchCSL(:,2);
patchleave_ms = dat.patchCSL(:,3);
rew_ms = dat.rew_ts;

sec1ix = 1000/tbin_ms;
sec2ix = 2000/tbin_ms;
times = -1000:tbin_ms:1000;

% Trial level features
patches = dat.patches;
patchCSL = dat.patchCSL;
prts = patchCSL(:,3) - patchCSL(:,2);
floor_prts = floor(prts);
patchType = patches(:,2);
rewsize = mod(patchType,10);

% make barcode matrices
nTimesteps = 15;
rew_barcode = zeros(length(patchCSL) , nTimesteps);
for iTrial = 1:length(patchCSL)
    rew_indices = round(rew_ms(rew_ms >= patchstop_ms(iTrial) & rew_ms < patchleave_ms(iTrial)) - patchstop_ms(iTrial)) + 1; 
    rew_barcode(iTrial,(max(rew_indices)+1):end) = -1;
    rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -2; % set part of patch after leave = -1
    rew_barcode(iTrial , rew_indices) = rewsize(iTrial);
end 

figure() 
sp_counter = 1;
for iRewsize = [1,2,4] 
    subplot(1,3,sp_counter)
    nil_trials = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == -1); 
    scatter(cs_ms(nil_trials),prts(nil_trials))
    title(sprintf("%i uL Trials Cue-stop time vs PRT",iRewsize)) 
    xlabel("Cue-stop duration") 
    ylabel("PRT") 
    sp_counter = sp_counter + 1;
end

% scatter(cs_ms(nil4_trials),prts(nil4_trials)) 
% title("Cue-stop time vs PRT") 
% xlabel("Cue-stop duration") 
% ylabel("PRT")

%% Perform our basic timewarping procedure  
tw_cueStop_frCell = {numel(sessions)};
for sIdx = 3:3 
    cs_lens = cellfun(@(x) size(x,2),cueStop_frCell{sIdx}); % to make our distn 
    median_len = median(cs_lens);  
    nNeurons = size(cueStop_frCell{sIdx}{1},1); 
    nTrials = length(cs_lens);
    tw_cueStop_frCell{sIdx} = nan(nNeurons,median_len,nTrials);
    
    % now linearly warp to the median t_len! 
    for iTrial = 1:nTrials
        tw_cueStop_frCell{sIdx}(:,:,iTrial) = imresize(cueStop_frCell{sIdx}{iTrial},[nNeurons,median_len]);
    end
    
end

%% Visualize warped and unwarped data  
close all
for sIdx = 3:3 
    for iTrial = 1:5 
        figure();colormap('jet')
        subplot(2,1,1) 
        imagesc(zscore(cueStop_frCell{sIdx}{iTrial},[],2)) 
        title("Un-warped Cue-Stop FR")
        subplot(2,1,2) 
        imagesc(zscore(squeeze(tw_cueStop_frCell{sIdx}(:,:,iTrial)),[],2)) 
        title("TimeWarped Cue-Stop FR")
    end
end
%% Now average over trials and make peak-sorted PETH  
peak_sorts = {numel(sessions)}; 
sorted_peths = {numel(sessions)};
for sIdx = 3:3 
    avg_cueStop_FR = mean(tw_cueStop_frCell{sIdx},3);
    normAvg_cueStop_FR = zscore(avg_cueStop_FR,[],2); 
    [~,max_ix] = max(normAvg_cueStop_FR,[],2); 
    [~,neuron_order] = sort(max_ix);
    sorted_peth = normAvg_cueStop_FR(neuron_order,:);  
    figure(); colormap('jet')
    imagesc(flipud(sorted_peth))   
    title("Peak-sorted PETH aligned to Cue")
    
    peak_sorts{sIdx} = neuron_order; 
    sorted_peths{sIdx} = sorted_peth; 
end

%% Now visualize some un-warped single trials with our ordering
for sIdx = 3:3
    for iTrial = 1:5
        figure();colormap('jet')
        imagesc(flipud(zscore(cueStop_frCell{sIdx}{iTrial}(peak_sorts{sIdx},:),[],2)))
        title("Un-warped Cue-Stop FR Sorted by Peak Location")
    end
end
