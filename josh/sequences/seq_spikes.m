%% code to plot rasters of sequence neuron spikes across trials 

paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_neuralData/neuroPixelsData/80';
paths.figs = '/Users/joshstern/Documents/UchidaLab_neuralData/neural_data_figs'; % where to save figs

addpath(genpath('/Users/joshstern/Documents/UchidaLab_neuralData/HGK_analysis_tools'));
addpath(genpath('/Users/joshstern/Documents/UchidaLab_neuralData'));

% analysis options
opt = struct;
opt.tbin = 0.02; % time bin for whole session rate matrix (in sec)
tbin_ms = opt.tbin*1000; % for making index vectors
opt.smoothSigma_time = 0.1 ; % 0.1; % gauss smoothing sigma for rate matrix (in sec)

sessions = dir(fullfile(paths.data,'*.mat'));
sessions = {sessions.name};

%% Extract FR matrices and timing information

FR_decVar = struct;
FRandTimes = struct;

for sIdx = 3:3 % 1:numel(sessions)
    session = sessions{sIdx}(1:end-4);
    fprintf('Loading session %d/%d: %s...\n',sIdx,numel(sessions),session);
    % load data
    dat = load(fullfile(paths.data,session));
    good_cells = dat.sp.cids(dat.sp.cgs==2); 
    
%     [~, spike_depths_all] = templatePositionsAmplitudes(dat.sp.temps, dat.sp.winv, dat.sp.ycoords, dat.sp.spikeTemplates, dat.sp.tempScalingAmps);
% 
%     % take median spike depth for each cell
%     spike_depths = nan(size(good_cells));
% 
%     parfor cIdx = 1:numel(good_cells)
%         spike_depths(cIdx) = median(spike_depths_all(dat.sp.clu==good_cells(cIdx)));
%     end  
%     
%     mm1_units = find(spike_depths > (max(spike_depths) - 1000));
%     
%     figure()
%     hist(spike_depths);
%     title("Distribution of spike depths")
    
    % time bins
    opt.tstart = 0;
    opt.tend = max(dat.sp.st);
    
    % behavioral events to align to
    patchcue_ms = dat.patchCSL(:,1)*1000;
    patchstop_ms = dat.patchCSL(:,2)*1000;
    patchleave_ms = dat.patchCSL(:,3)*1000;
    
    % Trial level features for decision variable creation
    patches = dat.patches;
    patchCSL = dat.patchCSL;

    new_fr_mat = true;
    if new_fr_mat == true
        % compute firing rate matrix
        tic
        [fr_mat, tbincent] = calcFRVsTime(good_cells,dat,opt); % calc from full matrix
        toc
    end 
    
%     fr_mat = fr_mat(mm1_units,:); 
%     display("Using units from top 1 mm of probe")

    buffer = 500; % buffer before leave in ms
    
    % create index vectors from our update timestamp vectors
    patchstop_ix = round(patchstop_ms / tbin_ms) + 1;
    patchleave_ix = min(round((patchleave_ms - buffer) / tbin_ms) + 1,size(fr_mat,2)); % might not be good
    
    % reinitialize ms vectors to make barcode matrix
    patchstop_ms = patchCSL(:,2);
    patchleave_ms = patchCSL(:,3);
    rew_ms = dat.rew_ts;
    rew_size = mod(dat.patches(:,2),10);
    prts = patchCSL(:,3) - patchCSL(:,2);
    floor_prts = floor(prts);
    patchType = patches(:,2);
    rewsize = mod(patchType,10);
    
    % make barcode matrices
    nTimesteps = 15;
    rew_barcode = zeros(length(patchCSL) , nTimesteps);
    for iTrial = 1:length(patchCSL)
        rew_indices = round(rew_ms(rew_ms >= patchstop_ms(iTrial) & rew_ms < patchleave_ms(iTrial)) - patchstop_ms(iTrial)) + 1;
        last_rew_ix = max(rew_indices);
        rew_sec_cell{iTrial} = rew_indices(rew_indices > 1);
        rew_barcode(iTrial , (last_rew_ix + 1):end) = -1; % set part of patch after last rew_ix = -1
        rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -2; % set part of patch after leave = -2
        rew_barcode(iTrial , rew_indices) = rewsize(iTrial);
    end
    
    % make struct
    FR_decVar(sIdx).fr_mat = {length(dat.patchCSL)};
    for iTrial = 1:length(dat.patchCSL)
        FR_decVar(sIdx).fr_mat{iTrial} = fr_mat(:,patchstop_ix(iTrial):patchleave_ix(iTrial));
        trial_len_ix = size(FR_decVar(sIdx).fr_mat{iTrial},2);
        FR_decVar(sIdx).decVarTime{iTrial} = (1:trial_len_ix) * tbin_ms / 1000;
        FR_decVar(sIdx).decVarTimeSinceRew{iTrial} = (1:trial_len_ix) * tbin_ms / 1000;
        
        for r = 1:numel(rew_sec_cell{iTrial})
            rew_ix = (rew_sec_cell{iTrial}(r) - 1) * 1000 / tbin_ms;
            FR_decVar(sIdx).decVarTimeSinceRew{iTrial}(rew_ix:end) =  (1:length(FR_decVar(sIdx).decVarTimeSinceRew{iTrial}(rew_ix:end))) * tbin_ms / 1000;
        end
    end
    
    figure();hold on;
    plot(FR_decVar(sIdx).decVarTime{39})
    hold on
    plot(FR_decVar(sIdx).decVarTimeSinceRew{39})
    legend("Time","Time since last reward")
    title("Trial 39 decision variables")
    
    FRandTimes(sIdx).fr_mat = fr_mat;
    FRandTimes(sIdx).stop_leave_ms = [patchstop_ms patchleave_ms];
    FRandTimes(sIdx).stop_leave_ix = [patchstop_ix patchleave_ix];
end

%% Sort by all trials to get ordering

index_sort_all = {sIdx};
for sIdx = 3:3
    decVar_bins = linspace(0,2,41);
    opt.norm = "zscore";
    opt.trials = 'all';
    opt.suppressVis = false;
    dvar = "timesince";
    [sorted_peth,neuron_order,unsorted_peth] = peakSortPETH(FR_decVar(sIdx),dvar,decVar_bins,opt);
    index_sort_all{sIdx} = neuron_order;
end

%% Make new datstructures to look at spiking 
% singleCell_trialed_sts: cell array of nNeurons matrices w/ trial # and st relative
%                         to trial start
% trial_rasters: cell array of nTrials matrices w/ neuron # and st relative
%                to trial start

for sIdx = 3:3   
    % load in data
    dat = load(fullfile(paths.data,session)); 
    patches = dat.patches;
    patchCSL = dat.patchCSL;  
    good_cells = dat.sp.cids(dat.sp.cgs==2); % going to need to map back to this 
    nNeurons = numel(good_cells);
    
    % trial stuff
    patchstop_sec = patchCSL(:,2);
    patchleave_sec = patchCSL(:,3);
    rew_sec = dat.rew_ts;
    prts = patchCSL(:,3) - patchCSL(:,2);
    floor_prts = floor(prts);
    patchType = patches(:,2);
    rewsize = mod(patchType,10);  
    [~,rewsort] = sort(rewsize); 
    
    % sort by PRT within 10,11,20,22,40,44, order [10,20,40,11,22,44]
    prt_R0_sort = []; 
    prt_RR_sort = []; 
    for iRewsize = [1,2,4] 
        trialsr0x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) < 0 & prts > 2.55);
        trialsrrx = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & prts > 2.55); 

        % sort by PRTs within reward history condition
        [~,prtsr0_sort] = sort(prts(trialsr0x));
        prtsr0_sort = trialsr0x(prtsr0_sort);
        [~,prtsrr_sort] = sort(prts(trialsrrx)); 
        prtsrr_sort = trialsrrx(prtsrr_sort);
        prt_rew2_sort = [prtsr0_sort ; prtsrr_sort]; 
        
        prt_R0_sort = [prt_R0_sort;prtsr0_sort];
        prt_RR_sort = [prt_RR_sort;prtsrr_sort];
    end 
    
    prt_withinRew_sort = [prt_R0_sort ; prt_RR_sort];
    
    nTrials = numel(prts);  
    
    good_cells_sorted = good_cells(index_sort_all{sIdx});
    
%     % first make trial_rasters as this will be easier to visualize  
%     trial_rasters = {nTrials};
%     for iTrial = 1:nTrials
%         trial_spikes = {nNeurons}; % make cell array, then concat all in one step for speed 
%         trial_sts = dat.sp.st(dat.sp.st > patchstop_sec(iTrial) & dat.sp.st < patchleave_sec(iTrial)) - patchstop_sec(iTrial); 
%         trial_clu = dat.sp.clu(dat.sp.st > patchstop_sec(iTrial) & dat.sp.st < patchleave_sec(iTrial)); 
%         for iNeuron = 1:nNeurons
%             good_cells_ix = good_cells_sorted(iNeuron); 
%             i_spike_ts = unique(trial_sts(trial_clu == good_cells_ix));  
%             trial_spikes{iNeuron} = [iNeuron + zeros(numel(i_spike_ts),1) i_spike_ts];
%         end  
%         % add to our data structure
%         trial_rasters{iTrial} = cat(1,trial_spikes{:}); 
%         if mod(iTrial,10) == 0
%             fprintf("Finished trial %i \n",iTrial) 
%         end
%     end 
    
    % now make singleCell_trialed_sts 
    singleCell_trialed_sts = {nNeurons}; 
    singleCell_trialed_smoothed = {nNeurons};
    sec_before = 0; 
    sec_after = 2000 / 1000;
    for iNeuron = 1:nNeurons  
        neuron_spikes = {numel(prt_withinRew_sort)}; % make cell array, then concat all in one step for speed  
        smoothed = {numel(prt_withinRew_sort)};
        good_cells_ix = good_cells_sorted(iNeuron); 
        neuron_sts = dat.sp.st(dat.sp.clu == good_cells_ix); 
        neuron_clu = dat.sp.clu(dat.sp.clu == good_cells_ix); 
        for j = 1:numel(prt_withinRew_sort)  
            iTrial = prt_withinRew_sort(j);
            start = patchstop_sec(iTrial) - sec_before; 
            finish = patchstop_sec(iTrial) + sec_after;
            i_spike_ts = unique(neuron_sts(neuron_sts > start & neuron_sts < finish)) - patchstop_sec(iTrial);  
            neuron_spikes{j} = [j + zeros(numel(i_spike_ts),1) i_spike_ts];  
            
            smoothed{j} = FR_decVar(sIdx).fr_mat{iTrial}(index_sort_all{sIdx}(iNeuron),1:round(sec_after * 1000 / tbin_ms));
        end
        % add to our data structure
        singleCell_trialed_sts{iNeuron} = cat(1,neuron_spikes{:});  
        singleCell_trialed_smoothed{iNeuron} = cat(1,smoothed{:});
        if mod(iNeuron,50) == 0
            fprintf("Finished neuron %i \n",iNeuron) 
        end
    end
end 

%% visualize some single trials
colors = cool(3); 
close all
for iTrial = 10:20
    iTrial_rew_ts = [1; 1000 * (rew_sec(rew_sec > patchstop_sec(iTrial) & rew_sec < patchleave_sec(iTrial)) - patchstop_sec(iTrial))]; 
    n_rews = numel(iTrial_rew_ts);
    figure()
    scatter(trial_rasters{iTrial}(:,2) * 1000,trial_rasters{iTrial}(:,1),1.5,'k.') 
    title(sprintf("Trial %i Raster",iTrial)) 
    xlim([0,max(trial_rasters{iTrial}(:,2) * 1000)]);ylim([0,max(trial_rasters{iTrial}(:,1))]) 
    xlabel("Time (msec)") 
    ylabel("Neuron Spiking Activity")   
    % plot rewards 
    hold on
    plot([iTrial_rew_ts iTrial_rew_ts]',repmat([1 nNeurons],[n_rews,1])','--','color',colors(3,:),'linewidth',1.5) 
    
end 

%% visualize some single cells   

colors3rew = [0 1 1 ;.5 .5 1;1 0 1;.5 1 1 ; .75 .75 1 ; 1 .5 1];
rewtrials = fliplr(cumsum([1 numel(find(rewsize == 1)) numel(find(rewsize == 2)) numel(find(rewsize == 4))])); 
trials10x = find(rew_barcode(:,1) == 1 & rew_barcode(:,2) < 0 & prts > 2.55);
trials20x = find(rew_barcode(:,1) == 2 & rew_barcode(:,2) < 0 & prts > 2.55);
trials40x = find(rew_barcode(:,1) == 4 & rew_barcode(:,2) < 0 & prts > 2.55);
trials11x = find(rew_barcode(:,1) == 1 & rew_barcode(:,2) == 1 & prts > 2.55);
trials22x = find(rew_barcode(:,1) == 2 & rew_barcode(:,2) == 2 & prts > 2.55);
trials44x = find(rew_barcode(:,1) == 4 & rew_barcode(:,2) == 4 & prts > 2.55);
r0trials = fliplr(cumsum([1 numel(trials10x) numel(trials20x) numel(trials40x) numel(trials11x) numel(trials22x) numel(trials44x)]));
close all  
for iNeuron = 60:70
    figure()  
    subplot(2,1,1)
    scatter(singleCell_trialed_sts{iNeuron}(:,2) * 1000,singleCell_trialed_sts{iNeuron}(:,1),10,'k.')   
    title(sprintf("Neuron %i Spiking Activity",iNeuron))
    xlabel("Time (msec)")    
    ylabel("Trials") 
%     yticks([])  
    hold on
    % shade in raster by reward size  
    for rewcounter = 1:6
        v = [1 r0trials(rewcounter); 1 r0trials(rewcounter+1);2000 r0trials(rewcounter+1); 2000 r0trials(rewcounter)];
        f = [1 2 3 4];
        patch('Faces',f,'Vertices',v,'FaceColor',colors3rew(rewcounter,:),'FaceAlpha',.1,'EdgeColor','none');  
    end 
    xlim([1 2000]) 
    ylim([1 numel(prt_withinRew_sort)])
    subplot(2,1,2) 
    imagesc(flipud(zscore(singleCell_trialed_smoothed{iNeuron},[],2)));colormap('jet')  
    title(sprintf("Neuron %i Z-Scored Activity",iNeuron))
    xlabel("Time (msec)")  
    xticks([12 25 37 50 62 75 87 100]);xticklabels([250 500 750 1000 1250 1500 1750 2000]) 
%     yticks([]) 
    ylabel("Trials") 
    colorbar()
end