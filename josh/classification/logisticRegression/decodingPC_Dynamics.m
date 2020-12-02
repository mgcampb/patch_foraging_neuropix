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

% load('forward_search_struct.mat') % struct from logistic regression forward search  

%% Extract FR matrices and timing information, get peaksort ordering
FR_decVar = struct; 
FRandTimes = struct;
index_sort_all = cell(numel(sessions),1);
for sIdx = 24:24
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

%% Make RX data structs
RX_data = cell(numel(sessions),1); 
RXNil_data = cell(numel(sessions),6);

for i = 1:numel(mPFC_sessions) 
    sIdx = mPFC_sessions(i);
    RX_data{sIdx} = struct; 
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    session = erase(sessions{sIdx}(1:end-4),'_'); % latex thing
    
    % reinitialize ms vectors
    patchstop_ms = data.patchCSL(:,2);
    patchleave_ms = data.patchCSL(:,3);
    prts = patchleave_ms - patchstop_ms;
    
    sec1ix = 1000/tbin_ms;
    sec2ix = 2000/tbin_ms; 
    sec3ix = 3000/tbin_ms;
    
    rew_barcode = rew_barcodes{sIdx};
  
    % RX: get that money
    rew_counter = 1;
    for iRewsize = [1,2,4]
        trials10x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == 0 & prts > 2.55);
        trials11x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & prts > 2.55);
        
        if ~isempty(trials10x) 
            tmp_trialsVis_cell = cellfun(@(x) x(:,1:sec2ix),classification_struct(sIdx).PCs(trials10x),'UniformOutput',false);
            RX_data{sIdx}(rew_counter).pc_mat = mean(cat(3,tmp_trialsVis_cell{:}),3);  
%             
% %             tmp_fr_mat = {length(trials10x)};
%             tmp_pc_mat = {length(trials10x)};
%             for j = 1:numel(trials10x)
%                 iTrial = trials10x(j);
% %                 stop_ix = FRandTimes(sIdx).stop_leave_ix(iTrial,1);
%                 tmp_pc_mat{j} = classification_struct(sIdx).PCs{iTrial}(:,1:sec2ix);
%             end
% %             mean_condition_fr = mean(cat(3,tmp_fr_mat{:}),3); % concatenate in third dimension, average over it
% %             RX_data{sIdx}(rew_counter).fr_mat = zscore(mean_condition_fr(index_sort_all{sIdx},:),[],2);
%             RX_data{sIdx}(rew_counter).pc_mat = mean(cat(3,tmp_pc_mat{:}),3);
        end
        
        if ~isempty(trials11x)
            tmp_fr_mat = {length(trials11x)};
            tmp_pc_mat = {length(trials11x)};
            for j = 1:numel(trials11x)
                iTrial = trials11x(j);
%                 stop_ix = FRandTimes(sIdx).stop_leave_ix(iTrial,1);
%                 tmp_fr_mat{j} = FRandTimes(sIdx).fr_mat(:,stop_ix:(stop_ix + sec2ix-1));
                tmp_pc_mat{j} = classification_struct(sIdx).PCs{iTrial}(:,1:sec2ix);
            end
%             mean_condition_fr = mean(cat(3,tmp_fr_mat{:}),3); % concatenate in third dimension, average over it
%             RX_data{sIdx}(rew_counter+3).fr_mat = zscore(mean_condition_fr(index_sort_all{sIdx},:),[],2); % 3 reward sizes
            RX_data{sIdx}(rew_counter+3).pc_mat = mean(cat(3,tmp_pc_mat{:}),3); 
            
        end
        
        % now do Nil Trials %%%
        trialsR0Nil = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) < 0 & prts > 1.55);
        trialsRRNil = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & rew_barcode(:,3) < 0 & prts > 1.55);
        
        % Get median PRTs - 1 to stretch to 
        medianPRT_R0Nil_ix = round((1000 * median(prts(trialsR0Nil) - 1) / tbin_ms)); 
        medianPostRewRT_RRNil_ix = round((1000 * (median(prts(trialsRRNil) - 1)) / tbin_ms)); 
        
        % collect stretched firing rates in cell 
        % note fancy stuff: only stretch following last rew reception 
        % also do this for R0Nil to match part of trial that is stretched
%         R0Nil_tmpCell = cellfun(@(x) imresize(x,[nNeurons,medianPRT_R0Nil_ix]),FR_decVar(sIdx).fr_mat(trialsR0Nil),'UniformOutput',false);
        R0Nil_tmpCell = cellfun(@(x) cat(2,x(:,1:sec1ix),imresize(x(:,sec1ix:end),[10,medianPRT_R0Nil_ix])) ... 
                                                            ,classification_struct(sIdx).PCs(trialsR0Nil),'UniformOutput',false);
        RRNil_tmpCell = cellfun(@(x) cat(2,x(:,1:sec1ix),imresize(x(:,sec1ix:end),[10,medianPostRewRT_RRNil_ix])) ... 
                                                            ,classification_struct(sIdx).PCs(trialsRRNil),'UniformOutput',false);
        RXNil_data{sIdx,rew_counter} = mean(cat(3,R0Nil_tmpCell{:}),3); 
        RXNil_data{sIdx,rew_counter + 3} = mean(cat(3,RRNil_tmpCell{:}),3);                                       
        rew_counter = rew_counter + 1; 
    end
end

%% Check out RX PC trajectories / PETHs   
% this isn't all that pretty 

close all
labels = {"10","20","40","11","22","44"};

for sIdx = 25
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
conds = 4:6;
% run PCA logreg beforehand to get meshgrid data
for sIdx = 15:18
%     disp(sessions(sIdx))  
    session = sessions{sIdx}(1:end-4); 
    session_title = ['m' session(1:2) ' ' session(end-2) '/' session([end-1:end])]; 
    decoding_order = forward_search(sIdx).pc_decodingOrder;
    all_concat_PCs_noPreRew = horzcat(classification_struct(sIdx).PCs_noPreRew{:})'; 
    all_concat_PCs_noPreRew = all_concat_PCs_noPreRew(:,decoding_order(1:2));
    all_concat_PCs = horzcat(classification_struct(sIdx).PCs{:})'; 
    all_concat_PCs = all_concat_PCs(:,decoding_order(1:2));
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
    pihat_mesh = mnrval(B,[x y]);
    figure();colormap("gray")
    scatter(x,y,[],pihat_mesh(:,2),'.');colorbar()
    title(sprintf("%s Top Decoding PC Dynamics over Regression Results",session_title))
    
    hold on 
    arrowSize = 5; 
    arrowGain = 0;
    arrowEdgeColor = 'k';
    % plot mean condition trajectories w.r.t. logistic regression meshgrid
    for condIdx = conds
        plot(RX_data{sIdx}(condIdx).pc_mat(decoding_order(1),:),RX_data{sIdx}(condIdx).pc_mat(decoding_order(2),:),'color',colors{condIdx},'linewidth',1.5)
%         sec_ticks = 50;  
        plot(RX_data{sIdx}(condIdx).pc_mat(decoding_order(1),50),RX_data{sIdx}(condIdx).pc_mat(decoding_order(2),50), 'kd', 'markerSize', 6, 'markerFaceColor',colors{condIdx});
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

%% Plot RX **stretched** trajectories on top of logistic regression meshgrid
close all
% colors = [.5 .5 .5;.75 .75 1;.5 .5 1;0 0 0; 1 .5 1 ;1 0 1];
colors = {[.5 1 1],[.75 .75 1],[1 .5 1],[0 1 1],[.5 .5 1],[1 0 1]}; 
conds = 1:3;
% run PCA logreg beforehand to get meshgrid data
for sIdx = 8 
%     disp(sessions(sIdx))  
    session = sessions{sIdx}(1:end-4); 
    session_title = ['m' session(1:2) ' ' session(end-2) '/' session([end-1:end])]; 
    decoding_order = forward_search(sIdx).pc_decodingOrder;
    all_concat_PCs_noPreRew = horzcat(classification_struct(sIdx).PCs_noPreRew{:})'; 
    all_concat_PCs_noPreRew = all_concat_PCs_noPreRew(:,decoding_order(1:2));
    all_concat_PCs = horzcat(classification_struct(sIdx).PCs{:})'; 
    all_concat_PCs = all_concat_PCs(:,decoding_order(1:2));
    session_len = size(all_concat_PCs,1);
    all_concat_labels_noPreRew = horzcat(classification_struct(sIdx).labels_noPreRew{:}) + 1;
    [B,dev,stats] = mnrfit(all_concat_PCs_noPreRew,all_concat_labels_noPreRew);
    pi_hat = mnrval(B,all_concat_PCs);

    % kinda jank, but just getting the x and y limits for visualization
    max_x = -inf; 
    max_y = -inf;
    for condIdx = conds 
        if max(abs(RXNil_data{sIdx,condIdx}(decoding_order(1),:))) > max_x
            max_x = max(abs(RXNil_data{sIdx,condIdx}(:)));  
        end
        if max(abs(RXNil_data{sIdx,condIdx}(decoding_order(2),:))) > max_y
            max_y = max(abs(RXNil_data{sIdx,condIdx}(:))); 
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
    pihat_mesh = mnrval(B,[x y]);
    figure();colormap("gray")
    scatter(x,y,[],pihat_mesh(:,2),'.');colorbar()
    title(sprintf("%s Top Decoding PC Dynamics over Regression Results",session_title))
    
    hold on 
    % add arrows
    arrowSize = 5; 
    arrowGain = 0;
    arrowEdgeColor = 'k'; 
    for cIdx = conds
        plot(RXNil_data{sIdx,cIdx}(decoding_order(1),:),RXNil_data{sIdx,cIdx}(decoding_order(2),:),'linewidth',2,'color',colors{cIdx}) 
        % plot x's at second marks
        sec_ticks = 50:50:size(RXNil_data{sIdx,cIdx},2);  
        plot(RXNil_data{sIdx,cIdx}(decoding_order(1),sec_ticks),RXNil_data{sIdx,cIdx}(decoding_order(2),sec_ticks), 'kd', 'markerSize', 6, 'markerFaceColor',colors{cIdx});
        
        % plot first point as dot
        plot(RXNil_data{sIdx,cIdx}(decoding_order(1),1),RXNil_data{sIdx,cIdx}(decoding_order(2),1), 'ko', 'markerSize', 6, 'markerFaceColor',colors{cIdx});
        % last point as arrow
        penultimatePoint = [RXNil_data{sIdx,cIdx}(decoding_order(1),end-1), RXNil_data{sIdx,cIdx}(decoding_order(2),end-1)];
        lastPoint = [RXNil_data{sIdx,cIdx}(decoding_order(1),end), RXNil_data{sIdx,cIdx}(decoding_order(2),end)];
        vel = norm(lastPoint - penultimatePoint);
%         xl = xlim();
%         yl = ylim();
        axLim = [xl yl];
        aSize = arrowSize + arrowGain * vel;  % if asked (e.g. for movies) arrow size may grow with vel
        arrowMMC(penultimatePoint, lastPoint, [], aSize, axLim, colors{cIdx}, arrowEdgeColor);
    end 
    xlabel(sprintf("PC %i",decoding_order(1)))
    ylabel(sprintf("PC %i",decoding_order(2))) 
    xlim(xl)
    ylim(yl)
end
