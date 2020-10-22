function [FR_decVar,FRandTimes] = genSeqStructs(paths,sessions,frCalc_opt,sIdx,buffer)
% Just a function to make two standardized structs useful for sequence
% analysis

    % initialize structs
    FR_decVar = struct;
    FRandTimes = struct;
    session = sessions{sIdx}(1:end-4);
    tbin_ms = frCalc_opt.tbin*1000;
    
    % load data
    dat = load(fullfile(paths.data,session));
    fprintf('Loading session %d/%d: %s...\n',sIdx,numel(sessions),session);
    good_cells = dat.sp.cids(dat.sp.cgs==2);  
    FR_decVar.goodcell_IDs = good_cells;
    
%      % todo: depth/brain region specificity
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
    frCalc_opt.tstart = 0;
    frCalc_opt.tend = max(dat.sp.st);
    
    % behavioral events to align to
    patchstop_ms = dat.patchCSL(:,2)*1000;
    patchleave_ms = dat.patchCSL(:,3)*1000;
    
    % Trial level features for decision variable creation
    patches = dat.patches;
    patchCSL = dat.patchCSL;

    new_fr_mat = true;
    if new_fr_mat == true
        % compute firing rate matrix
        tic
        [fr_mat, ~] = calcFRVsTime(good_cells,dat,frCalc_opt); % calc from full matrix
        toc
    end 
    
%     fr_mat = fr_mat(mm1_units,:); % subselect 
%     display("Using units from top 1 mm of probe")
    
    % create index vectors from our update timestamp vectors
    patchstop_ix = round(patchstop_ms / tbin_ms) + 1;
    patchleave_ix = min(round((patchleave_ms - buffer) / tbin_ms) + 1,size(fr_mat,2)); % might not be good
    
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
    FR_decVar.fr_mat = {length(dat.patchCSL)};
    for iTrial = 1:length(dat.patchCSL)
        FR_decVar.fr_mat{iTrial} = fr_mat(:,patchstop_ix(iTrial):patchleave_ix(iTrial));
        trial_len_ix = size(FR_decVar.fr_mat{iTrial},2);
        FR_decVar.decVarTime{iTrial} = (1:trial_len_ix) * tbin_ms / 1000;
        FR_decVar.decVarTimeSinceRew{iTrial} = (1:trial_len_ix) * tbin_ms / 1000;
        
        for r = 1:numel(rew_sec_cell{iTrial})
            rew_ix = (rew_sec_cell{iTrial}(r) - 1) * 1000 / tbin_ms;
            FR_decVar.decVarTimeSinceRew{iTrial}(rew_ix:end) =  (1:length(FR_decVar.decVarTimeSinceRew{iTrial}(rew_ix:end))) * tbin_ms / 1000;
        end
    end 
    
    % add PCA
    fr_mat_onPatch = horzcat(FR_decVar.fr_mat{:}); 
    fr_mat_onPatchZscore = zscore(fr_mat_onPatch,[],2); 
    [~,score,~,~,expl] = pca(fr_mat_onPatchZscore');
    score = score'; % reduced data  
    t_lens = cellfun(@(x) size(x,2),FR_decVar.fr_mat); 
    new_patchleave_ix = cumsum(t_lens);
    new_patchstop_ix = new_patchleave_ix - t_lens + 1;  
    FR_decVar.pca = {length(dat.patchCSL)};
    for iTrial = 1:length(dat.patchCSL) 
        FR_decVar.pca{iTrial} = score(1:10,new_patchstop_ix(iTrial):new_patchleave_ix(iTrial)); 
    end 
    FR_decVar.expl10 = sum(expl(1:10)) / sum(expl);

    FRandTimes.fr_mat = fr_mat;
    FRandTimes.stop_leave_ms = [patchstop_ms patchleave_ms];
    FRandTimes.stop_leave_ix = [patchstop_ix patchleave_ix];
end

