%% Analyze the low-dimensional trajectories and corresponding ridgeplots
%  - Assumes that we have already made forward_search struct from pca_logReg 
%  - Now that we have derived a low-dimensional basis that decodes leave
%    better than velocity, what does this low-dimensional activity look
%    like across days?  
%  - Basic question: is ramping overrepresented? sequentiality?

%% Basic path stuff + load forward search struct

paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice';
paths.figs = '/Users/joshstern/Documents/UchidaLab_NeuralData/neural_data_figs'; % where to save figs

addpath(genpath('/Users/joshstern/Documents/UchidaLab_NeuralData/HGK_analysis_tools'));
addpath(genpath('/Users/joshstern/Documents/UchidaLab_NeuralData'));

% analysis options
opt = struct;
opt.tbin = 0.02; % time bin for whole session rate matrix (in sec) 
tbin_ms = opt.tbin * 1000;
opt.smoothSigma_time = 0.100; % gauss smoothing sigma for rate matrix (in sec)

sessions = dir(fullfile(paths.data,'*.mat'));
sessions = {sessions.name};

load('forward_search_struct.mat') % struct from logistic regression forward search  

%% Extract FR matrices and timing information, get peaksort ordering
FR_decVar = struct; 
FRandTimes = struct;
index_sort_all = {sIdx};
for sIdx = 1:24
    buffer = 500;
    [FR_decVar_tmp,FRandTimes_tmp] = genSeqStructs(paths,sessions,opt,sIdx,buffer);
    % assign to sIdx 
    % for making peaksort PETHs / ordering
    FR_decVar(sIdx).fr_mat = FR_decVar_tmp.fr_mat;
    FR_decVar(sIdx).decVarTime = FR_decVar_tmp.decVarTime;
    FR_decVar(sIdx).decVarTimeSinceRew = FR_decVar_tmp.decVarTimeSinceRew;  
    FR_decVar(sIdx).pca = FR_decVar_tmp.pca; 
    FR_decVar(sIdx).expl10 = FR_decVar_tmp.expl10;
    % for making RXX plots
    FRandTimes(sIdx).fr_mat = FRandTimes_tmp.fr_mat;
    FRandTimes(sIdx).stop_leave_ms = FRandTimes_tmp.stop_leave_ms;
    FRandTimes(sIdx).stop_leave_ix = FRandTimes_tmp.stop_leave_ix;

    % Sort by all trials to get ordering
    decVar_bins = linspace(0,2,41);
    opt.norm = "zscore";
    opt.trials = 'all';
    opt.suppressVis = false;
    dvar = "timesince";
    [sorted_peth,neuron_order,unsorted_peth] = peakSortPETH(FR_decVar(sIdx),dvar,decVar_bins,opt);
    index_sort_all{sIdx} = neuron_order; 

end

%% Stop-aligned analysis of top decoding PCs
close all
colors = [.75 .75 1;.5 .5 1;1 .5 1 ;1 0 1];
for sIdx = 2:24
    session = sessions{sIdx}(1:end-4); 
    session_title =  session([1:2,end-2:end]);
    data = load(fullfile(paths.data,session));
    
    % Trial level features
    patches = data.patches;
    patchCSL = data.patchCSL;
    patchstop_ms = data.patchCSL(:,2);
    patchleave_ms = data.patchCSL(:,3);
    rew_ms = data.rew_ts;
    prts = patchCSL(:,3) - patchCSL(:,2);
    floor_prts = floor(prts);
    patchType = patches(:,2);
    rewsize = mod(patchType,10);
    % make barcode matrices
    nTimesteps = 15;
    rew_barcode = zeros(length(patchCSL) , nTimesteps);
    for iTrial = 1:length(patchCSL)
        rew_indices = round(rew_ms(rew_ms >= patchstop_ms(iTrial) & rew_ms < patchleave_ms(iTrial)) - patchstop_ms(iTrial)) + 1;
        rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -1; % set part of patch after leave = -1
        rew_barcode(iTrial , rew_indices) = rewsize(iTrial);
    end
    
    % now get into the neural data
    n_trials = numel(FR_decVar(sIdx).pca);
    decoding_order = forward_search(sIdx).pc_decodingOrder;   
%     fprintf("%s PCs to surpass velocity classification fidelity: %i \n",session_title,forward_search(sIdx).surpass_vel_nPCs)
    sec2ix = 2000 / tbin_ms;  
    
    t_lens = cellfun(@(x) size(x,2),FR_decVar(sIdx).pca); 
    
    % a few potential trial sorts for visualization  
    [~,rew_sort] = sort(rewsize);  
    trial_sort_str = "Reward Size";
    
    % visualize across trials aligned to patch stop
    figure()
    pcPETH = {3};
    for i = 1:3
        iPC = decoding_order(i); 
        pcPETH{i} = nan(n_trials,sec2ix);
        for iTrial = 1:n_trials
            pcPETH{i}(iTrial,1:min(t_lens(iTrial),sec2ix)) = FR_decVar(sIdx).pca{iTrial}(iPC,1:min(t_lens(iTrial),sec2ix));
        end  
        subplot(2,3,i) 
        if trial_sort_str == "Session"
            imagesc(pcPETH{i})   
        elseif trial_sort_str == "Reward Size" 
            imagesc(pcPETH{i}(rew_sort,:))
        end
        
        ylabel(sprintf("Trials in %s Order",trial_sort_str)) 
        title(sprintf("PC %i",iPC)) 
        yticks([])  
        xticks([0,500,1000,1500,2000] / tbin_ms) 
        xticklabels([0,500,1000,1500,2000]) 
        xlabel("Time (msec)")
        
        subplot(4,3,i+6) 
        hold on
        % plot mean + sem across RX conditions  
        colorcount = 1;
        for iRewsize = [2,4]
            trials10x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == 0 & prts > 2.55);
            trials11x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & prts > 2.55);
            pcPETH_i = pcPETH{i};
            plot(mean(pcPETH_i(trials10x,:)),'linewidth',1.5,'color',colors(colorcount,:))
            plot(mean(pcPETH_i(trials11x,:)),'linewidth',1.5,'color',colors(colorcount + 1,:))
            xticks([0,500,1000,1500,2000] / tbin_ms)
            xticklabels([0,500,1000,1500,2000])
            xlabel("Time (msec)") 
            ylabel(sprintf("PC %i Activity",iPC))
            colorcount = colorcount + 2;
        end 
        
%         if i == 1 
%             legend("20","22","40","44")
%         end
    end 
    suptitle(sprintf("Session %s PCs Needed to surpass velocity classification fidelity: %i \n",session_title,forward_search(sIdx).surpass_vel_nPCs))
end

%% Leave-aligned analysis of top decoding PCs
close all
colors = [.5 .5 1;1 0 1];
for sIdx = flipud(1:24)
    session = sessions{sIdx}(1:end-4); 
    session_title =  session([1:2,end-2:end]);
    data = load(fullfile(paths.data,session));
    
    % Trial level features
    patches = data.patches;
    patchCSL = data.patchCSL;
    patchType = patches(:,2);
    rewsize = mod(patchType,10);

    % now get into the neural data
    n_trials = numel(FR_decVar(sIdx).pca);
    decoding_order = forward_search(sIdx).pc_decodingOrder;   
    fprintf("PCs to surpass velocity classification fidelity: %i \n",forward_search(sIdx).surpass_vel_nPCs)
    sec1ix = 1000 / tbin_ms;  
    
    t_lens = cellfun(@(x) size(x,2),FR_decVar(sIdx).pca); 
    
    % a few potential trial sorts for visualization  
    [~,rew_sort] = sort(rewsize);  
    trial_sort_str = "Reward Size";
    
    % visualize across trials aligned to patch stop
    figure()
    pcPETH = {3};
    for i = 1:3
        iPC = decoding_order(i); 
        pcPETH{i} = nan(n_trials,sec1ix);
        for iTrial = 1:n_trials
            pcPETH{i}(iTrial,1:size(FR_decVar(sIdx).pca{iTrial}(iPC,max(1,(end-sec1ix+1)):end),2)) = FR_decVar(sIdx).pca{iTrial}(iPC,max(1,(end-sec1ix+1)):end);
        end  
        subplot(2,3,i) 
        if trial_sort_str == "Session"
            imagesc(pcPETH{i})   
        elseif trial_sort_str == "Reward Size" 
            imagesc(pcPETH{i}(rew_sort,:))
        end
        
        ylabel(sprintf("Trials in %s Order",trial_sort_str)) 
        title(sprintf("PC %i",iPC)) 
        yticks([])  
        xticks([0,500,1000,1500,2000] / tbin_ms) 
        xticklabels([0,500,1000,1500,2000]) 
        xlabel("Time (msec)")
        
        subplot(4,3,i+6) 
        hold on
        % plot mean + sem across RX conditions  
        colorcount = 1;
        for iRewsize = [2,4]
            pcPETH_i = pcPETH{i};
            plot(nanmean(pcPETH_i(rewsize == iRewsize,:)),'linewidth',1.5,'color',colors(colorcount,:))
            xticks([0,500,1000,1500,2000] / tbin_ms)
            xticklabels([0,500,1000,1500,2000])
            xlabel("Time (msec)") 
            ylabel(sprintf("PC %i Activity",iPC))
            colorcount = colorcount + 1;
        end 
%         
%         if i == 1 
%             legend("2 uL","4 uL")
%         end
    end
    suptitle(sprintf("Session %s PCs Needed to surpass velocity classification fidelity: %i \n",session_title,forward_search(sIdx).surpass_vel_nPCs))
end

%% 
RX_data = {}; 
RXX_data = {};
%% Make RX and RXX data structs 

for sIdx = 1:24
    RX_data{sIdx} = struct; 
    RXX_data{sIdx} = struct;
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    session = erase(sessions{sIdx}(1:end-4),'_'); % latex thing
    
    % reinitialize ms vectors
    patchstop_ms = data.patchCSL(:,2);
    patchleave_ms = data.patchCSL(:,3);
    rew_ms = data.rew_ts;
    
    sec1ix = 1000/tbin_ms;
    sec2ix = 2000/tbin_ms; 
    sec3ix = 3000/tbin_ms;
    times = -1000:tbin_ms:1000;
    
    % Trial level features
    patches = data.patches;
    patchCSL = data.patchCSL;
    prts = patchCSL(:,3) - patchCSL(:,2);
    floor_prts = floor(prts);
    patchType = patches(:,2);
    rewsize = mod(patchType,10);
    
    % make barcode matrices
    nTimesteps = 15;
    rew_barcode = zeros(length(patchCSL) , nTimesteps);
    for iTrial = 1:length(patchCSL)
        rew_indices = round(rew_ms(rew_ms >= patchstop_ms(iTrial) & rew_ms < patchleave_ms(iTrial)) - patchstop_ms(iTrial)) + 1;
        rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -1; % set part of patch after leave = -1
        rew_barcode(iTrial , rew_indices) = rewsize(iTrial);
    end
  
    % RX: get that money
    rew_counter = 1;
    for iRewsize = [1,2,4]
        trials10x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == 0 & prts > 2.55);
        trials11x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & prts > 2.55);
        
        if ~isempty(trials10x)
            tmp_fr_mat = {length(trials10x)};
            tmp_pc_mat = {length(trials10x)};
            for j = 1:numel(trials10x)
                iTrial = trials10x(j);
                stop_ix = FRandTimes(sIdx).stop_leave_ix(iTrial,1);
                tmp_fr_mat{j} = FRandTimes(sIdx).fr_mat(:,(stop_ix):stop_ix + sec2ix-1);
                tmp_pc_mat{j} = FR_decVar(sIdx).pca{iTrial}(:,1:sec2ix);
            end
            mean_condition_fr = mean(cat(3,tmp_fr_mat{:}),3); % concatenate in third dimension, average over it
            RX_data{sIdx}(rew_counter).fr_mat = zscore(mean_condition_fr(index_sort_all{sIdx},:),[],2);
            RX_data{sIdx}(rew_counter).pc_mat = mean(cat(3,tmp_pc_mat{:}),3);
        end
        
        if ~isempty(trials11x)
            tmp_fr_mat = {length(trials11x)};
            tmp_pc_mat = {length(trials11x)};
            for j = 1:numel(trials11x)
                iTrial = trials11x(j);
                stop_ix = FRandTimes(sIdx).stop_leave_ix(iTrial,1);
                tmp_fr_mat{j} = FRandTimes(sIdx).fr_mat(:,stop_ix:(stop_ix + sec2ix-1));
                tmp_pc_mat{j} = FR_decVar(sIdx).pca{iTrial}(:,1:sec2ix);
            end
            mean_condition_fr = mean(cat(3,tmp_fr_mat{:}),3); % concatenate in third dimension, average over it
            RX_data{sIdx}(rew_counter+3).fr_mat = zscore(mean_condition_fr(index_sort_all{sIdx},:),[],2); % 3 reward sizes
            RX_data{sIdx}(rew_counter+3).pc_mat = mean(cat(3,tmp_pc_mat{:}),3);
        end
        rew_counter = rew_counter + 1;
    end
    
%     % Now do the same for RXX 
%     rew_counter = 1;
%     for iRewsize = [2,4]
%         trials100x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == 0 & rew_barcode(:,3) == 0 & prts > 3.5);
%         trials110x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & rew_barcode(:,3) == 0 & prts > 3.5);
%         trials101x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == 0 & rew_barcode(:,3) == iRewsize & prts > 3.5);
%         trials111x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & rew_barcode(:,3) == iRewsize & prts > 3.5);
% 
%         tmp_fr_mat = {length(trials100x)}; 
%         tmp_pc_mat = {length(trials100x)};
%         for j = 1:numel(trials100x)
%             iTrial = trials100x(j);
%             stop_ix = FRandTimes(sIdx).stop_leave_ix(iTrial,1);
%             tmp_fr_mat{j} = FRandTimes(sIdx).fr_mat(:,(stop_ix):stop_ix + sec2ix-1);
%             tmp_pc_mat{j} = FR_decVar(sIdx).pca{iTrial}(:,1:sec2ix);
%         end
%         mean_condition_fr = mean(cat(3,tmp_fr_mat{:}),3); % concatenate in third dimension, average over it
%         RXX_data{sIdx}(rew_counter).fr_mat = zscore(mean_condition_fr(index_sort_all{sIdx},:),[],2); 
%         RXX_data{sIdx}(rew_counter).pc_mat = mean(cat(3,tmp_pc_mat{:}),3);
%         
%         tmp_fr_mat = {length(trials110x)}; 
%         tmp_pc_mat = {length(trials110x)};
%         for j = 1:numel(trials110x)
%             iTrial = trials110x(j);
%             stop_ix = FRandTimes(sIdx).stop_leave_ix(iTrial,1);
%             tmp_fr_mat{j} = FRandTimes(sIdx).fr_mat(:,(stop_ix):stop_ix + sec2ix-1);
%             tmp_pc_mat{j} = FR_decVar(sIdx).pca{iTrial}(:,1:sec2ix);
%         end
%         mean_condition_fr = mean(cat(3,tmp_fr_mat{:}),3); % concatenate in third dimension, average over it
%         RXX_data{sIdx}(rew_counter+1).fr_mat = zscore(mean_condition_fr(index_sort_all{sIdx},:),[],2); 
%         RXX_data{sIdx}(rew_counter+1).pc_mat = mean(cat(3,tmp_pc_mat{:}),3);
%         
%         tmp_fr_mat = {length(trials101x)}; 
%         tmp_pc_mat = {length(trials101x)};
%         for j = 1:numel(trials101x)
%             iTrial = trials101x(j);
%             stop_ix = FRandTimes(sIdx).stop_leave_ix(iTrial,1);
%             tmp_fr_mat{j} = FRandTimes(sIdx).fr_mat(:,(stop_ix):stop_ix + sec2ix-1);
%             tmp_pc_mat{j} = FR_decVar(sIdx).pca{iTrial}(:,1:sec2ix);
%         end
%         mean_condition_fr = mean(cat(3,tmp_fr_mat{:}),3); % concatenate in third dimension, average over it
%         RXX_data{sIdx}(rew_counter+2).fr_mat = zscore(mean_condition_fr(index_sort_all{sIdx},:),[],2); 
%         RXX_data{sIdx}(rew_counter+2).pc_mat = mean(cat(3,tmp_pc_mat{:}),3);
%         
%         tmp_fr_mat = {length(trials111x)}; 
%         tmp_pc_mat = {length(trials111x)};
%         for j = 1:numel(trials111x)
%             iTrial = trials111x(j);
%             stop_ix = FRandTimes(sIdx).stop_leave_ix(iTrial,1);
%             tmp_fr_mat{j} = FRandTimes(sIdx).fr_mat(:,(stop_ix):stop_ix + sec2ix-1);
%             tmp_pc_mat{j} = FR_decVar(sIdx).pca{iTrial}(:,1:sec2ix);
%         end
%         mean_condition_fr = mean(cat(3,tmp_fr_mat{:}),3); % concatenate in third dimension, average over it
%         RXX_data{sIdx}(rew_counter+3).fr_mat = zscore(mean_condition_fr(index_sort_all{sIdx},:),[],2); 
%         RXX_data{sIdx}(rew_counter+3).pc_mat = mean(cat(3,tmp_pc_mat{:}),3);
%         
%         rew_counter = rew_counter + 4; 
%     end
end

%% Check out RX PC trajectories / PETHs   
% this isn't all that pretty 

close all
labels = {"10","20","40","11","22","44"};

for sIdx = 18:22
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    
    % Trial level features
    patches = data.patches;
    patchCSL = data.patchCSL;
    patchType = patches(:,2);
    rewsize = mod(patchType,10);
    
    % now get into the neural data
    n_trials = numel(FR_decVar(sIdx).pca);
    decoding_order = forward_search(sIdx).pc_decodingOrder;
    fprintf("PCs to surpass velocity classification fidelity: %i \n",forward_search(sIdx).surpass_vel_nPCs)
    
    for condIdx = 1:6
        % show sorted PETH
        figure();colormap('jet');
        subplot(1,3,1)
        imagesc(flipud(RX_data{sIdx}(condIdx).fr_mat))
        xlabel("Time (ms)")
        xticks([0,50,100]) % just for the 2 sec data
        xticklabels([0,1000,2000])
        title(sprintf("%s PETH",labels{condIdx}))
        subplot(2,3,2)
        plot(RX_data{sIdx}(condIdx).pc_mat(1:2,:)','linewidth',1.5)
        title("Top PCs by Variance Explained")
        subplot(2,3,5)
        plot(RX_data{sIdx}(condIdx).pc_mat(decoding_order(1:2),:)','linewidth',1.5)
        title("Top PCs by Classification Fidelity")
        subplot(2,3,3)
        plot(RX_data{sIdx}(condIdx).pc_mat(1,:),RX_data{sIdx}(condIdx).pc_mat(2,:),'linewidth',1.5)
        title("Variance Explained PC1:2 Trajectories")
        subplot(2,3,6)
        plot(RX_data{sIdx}(condIdx).pc_mat(decoding_order(1),:),RX_data{sIdx}(condIdx).pc_mat(decoding_order(2),:),'linewidth',1.5)
        title("Classification Fidelity PC1:2 Trajectories")
    end
end 

%% Plot RX trajectories on top of logistic regression meshgrid
close all
% colors = [.5 .5 .5;.75 .75 1;.5 .5 1;0 0 0; 1 .5 1 ;1 0 1];
colors = {[.5 1 1],[.75 .75 1],[1 .5 1],[0 1 1],[.5 .5 1],[1 0 1]}; 
conds = [2,3,5,6];
% run PCA logreg beforehand to get meshgrid data
for sIdx = 17:17
    disp(sessions(sIdx))
    decoding_order = forward_search(sIdx).pc_decodingOrder;
    all_concat_PCs_noPreRew = horzcat(classification_struct(sIdx).PCs_noPreRew{:})';
    all_concat_PCs = horzcat(classification_struct(sIdx).PCs{:})';
    session_len = size(all_concat_PCs,1);
    all_concat_labels_noPreRew = horzcat(classification_struct(sIdx).labels_noPreRew{:}) + 1;
    [B,dev,stats] = mnrfit(all_concat_PCs_noPreRew,all_concat_labels_noPreRew);
    pi_hat = mnrval(B,all_concat_PCs);

    % kinda jank, but just getting the x and y limits for visualization
    max_x = -inf; 
    max_y = -inf;
    for condIdx = conds 
        if max(abs(RX_data{sIdx}(condIdx).pc_mat(decoding_order(1),:))) > max_x
            max_x = max(abs(RX_data{sIdx}(condIdx).pc_mat(decoding_order(1),:)));  
        end
        if max(abs(RX_data{sIdx}(condIdx).pc_mat(decoding_order(2),:))) > max_y
            max_y = max(abs(RX_data{sIdx}(condIdx).pc_mat(decoding_order(2),:))); 
        end
    end    
    pad = 3; 
    max_x = max_x + pad; 
    max_y = max_y + pad;
    xl = [-max_x max_x]; 
    yl = [-max_y max_y];
    
    % show results using meshgrid
    [x,y] = meshgrid(xl(1):.05:xl(2),yl(1):.05:yl(2));
    x = x(:);
    y = y(:);
    % this is currently not programatic! 
    pihat_mesh = mnrval(B,[y zeros(size(x,1),2) x zeros(size(x,1),6)]);
%     pihat_mesh = mnrval(B,[x zeros(size(x,1),6) y zeros(size(x,1),2)]);
    figure();colormap("gray")
    scatter(x,y,[],pihat_mesh(:,2),'.');colorbar()
    %     xlabel(sprintf("PC%i",decode_pc1));ylabel(sprintf("PC%i",decode_pc2))
    title("Logistic Regression Results as Meshgrid")
    
    hold on 
    arrowSize = 5; 
    arrowGain = 0;
    arrowEdgeColor = 'k';
    % plot mean condition trajectories w.r.t. logistic regression meshgrid
    for condIdx = conds
        plot(RX_data{sIdx}(condIdx).pc_mat(decoding_order(1),:),RX_data{sIdx}(condIdx).pc_mat(decoding_order(2),:),'color',colors{condIdx},'linewidth',1.5)
        
        plot(RX_data{sIdx}(condIdx).pc_mat(decoding_order(1),1), RX_data{sIdx}(condIdx).pc_mat(decoding_order(2),1), 'ko', 'markerSize', 6, 'markerFaceColor',colors{condIdx});
        % for arrow, figure out last two points, and (if asked) supress the arrow if velocity is
        % below a threshold.
        penultimatePoint = [RX_data{sIdx}(condIdx).pc_mat(decoding_order(1),end-1), RX_data{sIdx}(condIdx).pc_mat(decoding_order(2),end-1)];
        lastPoint = [RX_data{sIdx}(condIdx).pc_mat(decoding_order(1),end), RX_data{sIdx}(condIdx).pc_mat(decoding_order(2),end)];
        vel = norm(lastPoint - penultimatePoint);
        
        axLim = [xl yl];
        aSize = arrowSize + arrowGain * vel;  % if asked (e.g. for movies) arrow size may grow with vel
        arrowMMC(penultimatePoint, lastPoint, [], aSize, axLim, colors{condIdx}, arrowEdgeColor);
        
    end
    xlim(xl)
    ylim(yl) 
    xlabel(sprintf("PC %i",decoding_order(1)))
    ylabel(sprintf("PC %i",decoding_order(2)))
end

%% Same for RXX
close all
% colors = [.5 .5 .5;.75 .75 1;.5 .5 1;0 0 0; 1 .5 1 ;1 0 1];
colors = {[.5 1 1],[.75 .75 1],[1 .5 1],[0 1 1],[.5 .5 1],[1 0 1]}; 
conds = [2,3,5,6];
% run PCA logreg beforehand to get meshgrid data
for sIdx = 24:24
    
    all_concat_PCs_noPreRew = horzcat(classification_struct(sIdx).PCs_noPreRew{:})';
    all_concat_PCs = horzcat(classification_struct(sIdx).PCs{:})';
    session_len = size(all_concat_PCs,1);
    all_concat_labels_noPreRew = horzcat(classification_struct(sIdx).labels_noPreRew{:}) + 1;
    [B,dev,stats] = mnrfit(all_concat_PCs_noPreRew,all_concat_labels_noPreRew);
    pi_hat = mnrval(B,all_concat_PCs);

    % kinda jank, but just getting the x and y limits for visualization
    max_x = -inf; 
    max_y = -inf;
    for condIdx = conds 
        if max(abs(RX_data{sIdx}(condIdx).pc_mat(decoding_order(1),:))) > max_x
            max_x = max(abs(RX_data{sIdx}(condIdx).pc_mat(decoding_order(1),:)));  
        end
        if max(abs(RX_data{sIdx}(condIdx).pc_mat(decoding_order(2),:))) > max_y
            max_y = max(abs(RX_data{sIdx}(condIdx).pc_mat(decoding_order(2),:))); 
        end
    end    
    pad = 3; 
    max_x = max_x + pad; 
    max_y = max_y + pad;
    xl = [-max_x max_x]; 
    yl = [-max_y max_y];
    
    % show results using meshgrid
    [x,y] = meshgrid(xl(1):.05:xl(2),yl(1):.05:yl(2));
    x = x(:);
    y = y(:);
    % this is currently not programatic!
    pihat_mesh = mnrval(B,[x zeros(size(x,1),6) y zeros(size(x,1),2)]);
    figure();colormap("hot")
    scatter(x,y,[],pihat_mesh(:,2),'.');colorbar()
    %     xlabel(sprintf("PC%i",decode_pc1));ylabel(sprintf("PC%i",decode_pc2))
    title("Logistic Regression Results as Meshgrid")
    
    hold on 
    arrowSize = 5; 
    arrowGain = 0;
    arrowEdgeColor = 'k';
    % plot mean condition trajectories w.r.t. logistic regression meshgrid
    for condIdx = conds
        plot(RX_data{sIdx}(condIdx).pc_mat(decoding_order(1),:),RX_data{sIdx}(condIdx).pc_mat(decoding_order(2),:),'color',colors{condIdx},'linewidth',1.5)
        
        plot(RX_data{sIdx}(condIdx).pc_mat(decoding_order(1),1), RX_data{sIdx}(condIdx).pc_mat(decoding_order(2),1), 'ko', 'markerSize', 6, 'markerFaceColor',colors{condIdx});
        % for arrow, figure out last two points, and (if asked) supress the arrow if velocity is
        % below a threshold.
        penultimatePoint = [RX_data{sIdx}(condIdx).pc_mat(decoding_order(1),end-1), RX_data{sIdx}(condIdx).pc_mat(decoding_order(2),end-1)];
        lastPoint = [RX_data{sIdx}(condIdx).pc_mat(decoding_order(1),end), RX_data{sIdx}(condIdx).pc_mat(decoding_order(2),end)];
        vel = norm(lastPoint - penultimatePoint);
        
        axLim = [xl yl];
        aSize = arrowSize + arrowGain * vel;  % if asked (e.g. for movies) arrow size may grow with vel
        arrowMMC(penultimatePoint, lastPoint, [], aSize, axLim, colors{condIdx}, arrowEdgeColor);
        
    end
    xlim(xl)
    ylim(yl)
end