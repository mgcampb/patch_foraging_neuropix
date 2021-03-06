function [FR_decVar,FRandTimes] = genSeqStructs(paths,sessions,opt,sIdx)
% Just a function to make two standardized structs useful for sequence
% analysis
    
    % extract keyword arguments
    preLeave_buffer = 500;
    if exist('opt', 'var') && isfield(opt,'preLeave_buffer')
        preLeave_buffer = opt.preLeave_buffer;
    end  
    
    region_selection = []; 
    if exist('opt','var') && isfield(opt,'region_selection') 
        region_selection = opt.region_selection;  
    end 
    
    % initialize structs
    FR_decVar = struct;
    FRandTimes = struct;
    session = sessions{sIdx}(1:end-4);
    tbin_ms = opt.tbin*1000;
    
    % load data
    dat = load(fullfile(paths.data,session)); 
%     disp(dat)
    fprintf('Loading session %d/%d: %s...\n',sIdx,numel(sessions),session);
    good_cells = dat.sp.cids(dat.sp.cgs==2); 
    
    % subselect w/ rough brain region
    if ~isempty(region_selection) 
        good_cells = good_cells(dat.brain_region_rough == region_selection);
    end
    
    % now load ramp struct 
    ramp_fname = [paths.rampIDs '/m' sessions{sIdx}(1:end-4) '_rampIDs.mat'];   
    if exist(ramp_fname,'file')
        % add indices of ramping cells 
        ramp_file = load(ramp_fname); 
        ramps = ramp_file.ramps;  
        FR_decVar.ramp_up_all_ix = find(ismember(good_cells,ramps.up_all)); 
        FR_decVar.ramp_up_common_ix = find(ismember(good_cells,ramps.up_common)); 
    end

    FR_decVar.goodcell_IDs = good_cells; 
    [~, spike_depths_all] = templatePositionsAmplitudes(dat.sp.temps, dat.sp.winv, dat.sp.ycoords, dat.sp.spikeTemplates, dat.sp.tempScalingAmps);
    % take median spike depth for each cell
    spike_depths = nan(size(good_cells));

    for cIdx = 1:numel(good_cells)
        spike_depths(cIdx) = median(spike_depths_all(dat.sp.clu==good_cells(cIdx)));
    end   
    FR_decVar.spike_depths = spike_depths; 
    
    % time bins
    opt.tstart = 0;
    opt.tend = max(dat.sp.st);
    
    % behavioral events to align to
    patchstop_ms = dat.patchCSL(:,2)*1000;
    patchleave_ms = dat.patchCSL(:,3)*1000;
    
    % Trial level features for decision variable creation
    patches = dat.patches;
    patchCSL = dat.patchCSL; 
    nTrials = length(patchCSL);

    new_fr_mat = true;
    if new_fr_mat == true
        % compute firing rate matrix
        [fr_mat, ~] = calcFRVsTime(good_cells,dat,opt); % calc from full matrix
    end 

    % create index vectors from our update timestamp vectors
    patchstop_ix = round(patchstop_ms / tbin_ms) + 1;
    patchleave_ix = min(round((patchleave_ms - preLeave_buffer) / tbin_ms) + 1,size(fr_mat,2)); % might not be good
    
    % reinitialize ms vectors to make barcode matrix
    patchstop_ms = patchCSL(:,2);
    patchleave_ms = patchCSL(:,3);
    rew_ms = dat.rew_ts;
    prts = patchCSL(:,3) - patchCSL(:,2);
    floor_prts = floor(prts);
    patchType = patches(:,2);
    rewsize = mod(patchType,10);
    
    % make barcode matrices
    nTimesteps = 15;
    rew_barcode = zeros(nTrials , nTimesteps); 
    rew_sec_cell = cell(nTrials,1);
    for iTrial = 1:nTrials
        rew_indices = round(rew_ms(rew_ms >= patchstop_ms(iTrial) & rew_ms < patchleave_ms(iTrial)) - patchstop_ms(iTrial)) + 1;
        last_rew_ix = max(rew_indices);
        rew_sec_cell{iTrial} = rew_indices(rew_indices > 1);
        rew_barcode(iTrial , (last_rew_ix + 1):end) = -1; % set part of patch after last rew_ix = -1
        rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -2; % set part of patch after leave = -2
        rew_barcode(iTrial , rew_indices) = rewsize(iTrial);
    end
    
    % make struct
    FR_decVar.fr_mat = cell(length(dat.patchCSL),1); 
    FR_decVar.decVarTime = cell(length(dat.patchCSL),1);
    FR_decVar.decVarTimeSinceRew = cell(length(dat.patchCSL),1);
    FR_decVar.rew_sec = cell(length(dat.patchCSL),1);
    FR_decVar.rew_num = cell(length(dat.patchCSL),1);
    for iTrial = 1:length(dat.patchCSL)
        FR_decVar.fr_mat{iTrial} = fr_mat(:,patchstop_ix(iTrial):patchleave_ix(iTrial));
        trial_len_ix = size(FR_decVar.fr_mat{iTrial},2);
        FR_decVar.decVarTime{iTrial} = (1:trial_len_ix) * tbin_ms / 1000;
        FR_decVar.decVarTimeSinceRew{iTrial} = (1:trial_len_ix) * tbin_ms / 1000;  
        FR_decVar.rew_sec{iTrial} = zeros(1,trial_len_ix);
        FR_decVar.rew_num{iTrial} = zeros(1,trial_len_ix);
        
        for r = 1:numel(rew_sec_cell{iTrial})
            rew_ix = (rew_sec_cell{iTrial}(r) - 1) * 1000 / tbin_ms;
            FR_decVar.decVarTimeSinceRew{iTrial}(rew_ix:end) =  (1:length(FR_decVar.decVarTimeSinceRew{iTrial}(rew_ix:end))) * tbin_ms / 1000;
            FR_decVar.rew_sec{iTrial}(rew_ix:end) = rew_sec_cell{iTrial}(r); 
            FR_decVar.rew_num{iTrial}(rew_ix:end) = r; 
        end
    end 
    
%     % add PCA
%     fr_mat_onPatch = horzcat(FR_decVar.fr_mat{:}); 
%     fr_mat_onPatchZscore = zscore(fr_mat_onPatch,[],2); 
%     [~,score,~,~,expl] = pca(fr_mat_onPatchZscore');
%     score = score'; % reduced data  
%     t_lens = cellfun(@(x) size(x,2),FR_decVar.fr_mat); 
%     new_patchleave_ix = cumsum(t_lens);
%     new_patchstop_ix = new_patchleave_ix - t_lens + 1;  
%     FR_decVar.pca = {length(dat.patchCSL)};
%     for iTrial = 1:length(dat.patchCSL) 
%         FR_decVar.pca{iTrial} = score(1:10,new_patchstop_ix(iTrial):new_patchleave_ix(iTrial)); 
%     end 
%     FR_decVar.expl10 = sum(expl(1:10)) / sum(expl);

    FRandTimes.fr_mat = fr_mat;
    FRandTimes.stop_leave_ms = [patchstop_ms patchleave_ms];
    FRandTimes.stop_leave_ix = [patchstop_ix patchleave_ix];
end

