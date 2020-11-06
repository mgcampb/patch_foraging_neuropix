%% Find transiently responsive neurons w/ laura driscoll's method 

paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice';
paths.figs = '/Users/joshstern/Documents/UchidaLab_NeuralData/neural_data_figs'; % where to save figs

addpath(genpath('/Users/joshstern/Documents/UchidaLab_NeuralData'));

% analysis options
opt = struct;
opt.tbin = 0.02; % time bin for whole session rate matrix (in sec) 
tbin_ms = opt.tbin * 1000;
opt.smoothSigma_time = 0.100; % gauss smoothing sigma for rate matrix (in sec) 
opt.preLeave_buffer = 500;

sessions = dir(fullfile(paths.data,'*.mat'));
sessions = {sessions.name}; 

%% Extract FR matrices and timing information 
FR_decVar = struct; 
FRandTimes = struct;
for sIdx = 22:24
    [FR_decVar_tmp,~] = genSeqStructs(paths,sessions,opt,sIdx);
    % assign to sIdx
    FR_decVar(sIdx).fr_mat = FR_decVar_tmp.fr_mat; 
    FR_decVar(sIdx).goodcell_IDs = FR_decVar_tmp.goodcell_IDs; 
    FR_decVar(sIdx).decVarTime = FR_decVar_tmp.decVarTime;
    FR_decVar(sIdx).decVarTimeSinceRew = FR_decVar_tmp.decVarTimeSinceRew; 
    FR_decVar(sIdx).cell_depths = FR_decVar_tmp.spike_depths;
%     FRandTimes(sIdx).fr_mat = FRandTimes_tmp.fr_mat;
%     FRandTimes(sIdx).stop_leave_ms = FRandTimes_tmp.stop_leave_ms;
%     FRandTimes(sIdx).stop_leave_ix = FRandTimes_tmp.stop_leave_ix; 
end  

%% Bin activity and compare to shuffle (un-normalized) 
index_sort_all = cell(numel(sessions),1); 
decVar_bins = linspace(0,2,41);  
bin_tbin = decVar_bins(2) - decVar_bins(1);
peak_ix_cell = cell(numel(sessions),4); 
driscoll_midresp_struct = struct();

for sIdx = 24:24 
    session = sessions{sIdx}; 
    data = load(fullfile(paths.data,session));   
    rewsize = mod(data.patches(:,2),10);   
    nTrials = length(rewsize);
    
    % Get a consistent sort for visualization
    opt.norm = "zscore";
    opt.trials = 'all'; % NOTE THIS
    opt.suppressVis = true;
    dvar = "timesince";
    [sorted_peth,neuron_order,unsorted_peth] = peakSortPETH(FR_decVar(sIdx),dvar,decVar_bins,opt);
    index_sort_all{sIdx} = neuron_order;   
    
    for iRewsize = 2
        trial_selection = find(rewsize == iRewsize); 

        % Only select one reward size at a time
        fr_mat = FR_decVar(sIdx).fr_mat(trial_selection);
        fr_mat = cat(2,fr_mat{:}); 
        timesince = FR_decVar(sIdx).decVarTimeSinceRew(trial_selection);
        timesince = cat(2,timesince{:});    
        timepatch = FR_decVar(sIdx).decVarTime(trial_selection); 
        timepatch = cat(2,timepatch{:});   
        nNeurons = size(fr_mat,1); 
        
        % optionally take off some time before reward reception (smoothing)
        % would be nice to procedurally generate indices  
        preRew_buffer = true;
        if preRew_buffer == true
            preRew_buffer = round(opt.smoothSigma_time * 3 * 1000 / tbin_ms); 
            rew_ix = find(timesince == tbin_ms / 1000 & timepatch > tbin_ms / 1000);  
            preRew_bool = false(length(timesince));
            for iRew = 1:numel(rew_ix)  
                preRew_bool(rew_ix(iRew) - preRew_buffer:rew_ix(iRew)) = true;
            end 
            fr_mat(:,preRew_bool) = []; 
            timesince(preRew_bool) = []; 
            timepatch(preRew_bool) = [];
        end 
        
        % optionally take off some time after patch stop 
        postStop_buffer = true; 
        if postStop_buffer == true 
            postStop_buffer = round(500 / tbin_ms); 
            stop_ix = find(timepatch == tbin_ms / 1000); 
            postStop_bool = false(length(timesince));
            for iTrial = 1:numel(stop_ix) 
                postStop_bool(stop_ix(iTrial):stop_ix(iTrial) + postStop_buffer) = true; 
            end 
            fr_mat(:,postStop_bool) = []; 
            timesince(postStop_bool) = []; 
            timepatch(postStop_bool) = [];
        end
        
        % Bin activity and take mean per bin
        [~,~,bin] = histcounts(timesince,decVar_bins);
        unshuffled_peth = nan(nNeurons,max(bin));
        for i = 1:max(bin)
            unshuffled_peth(:,i) = mean(fr_mat(:,bin==i),2);
        end

        % Find shuffle distribution 
        nShuffles = 500;  
        new_shuffle = true; % for development
        if new_shuffle == true
            shuffle_greater = zeros(size(fr_mat,1),max(bin));
            for shuffle = 1:nShuffles  
                % shuffle by rotational shift 
                fr_shuffle = fr_mat;
                shifts = randi(size(fr_shuffle,2),nNeurons,1);
                parfor neuron = 1:nNeurons
                    fr_shuffle(neuron,:) = circshift(fr_shuffle(neuron,:),shifts(neuron));
                end

                % calculate average FR w.r.t. time since reward
                i_shufflePETH = nan(nNeurons,max(bin));
                for i = 1:max(bin)
                    i_shufflePETH(:,i) = mean(fr_shuffle(:,bin==i),2);
                end 

                % add 1 to the locations where the shuffled PETH was greater than unshuffled
                [r,c] = find(i_shufflePETH > unshuffled_peth);  
                ind = sub2ind(size(i_shufflePETH),r,c);
                shuffle_greater(ind) = shuffle_greater(ind) + 1;

                % track progress
                if mod(shuffle,100) == 0 
                    disp(shuffle)
                end
            end  
        end

        % now find p value per bin per neuron
        pValue_peth = shuffle_greater / nShuffles; 
        
        % find significant peaks
        [r,c] = find(pValue_peth < .01); 
        ind = sub2ind(size(pValue_peth),r,c); 
        significant_peaks = zeros(size(pValue_peth)); 
        significant_peaks(ind) = 1;   

        % perform convolution to find where we have structure (not just sig ix)
        peak_width = 3;
        C = conv2(significant_peaks,ones(1,peak_width),'same'); 
        % find peak indices
        peak_ix = nan(nNeurons,1); 
        doublepeak = false(nNeurons,1);
        for neuron = 1:nNeurons  
            neuron_sig_ix = find(C(neuron,:) >= peak_width); % find where we had significant peak in a row   
            % handle double peak
            if ~isempty(find(neuron_sig_ix(2:end) - neuron_sig_ix(1:end-1) > 1,1)) 
                doublepeak(neuron) = true;
            end 
            [~,neuron_peak_ix] = max(unshuffled_peth(neuron,neuron_sig_ix)); 
            if ~isempty(neuron_peak_ix)
                peak_ix(neuron) = neuron_sig_ix(neuron_peak_ix); 
            end
        end   
        
        [~,driscoll_sort] = sort(peak_ix);
        driscoll_sort = driscoll_sort(ismember(driscoll_sort,find(~isnan(peak_ix)))); % get rid of non significant cells
        % log information
        peak_ix_cell{sIdx,iRewsize} = peak_ix;  
        
        driscoll_midresp_struct(sIdx,iRewsize).peak_ix = peak_ix * bin_tbin;
        driscoll_midresp_struct(sIdx,iRewsize).midresp = find(peak_ix > 5 & peak_ix < 35); 
        driscoll_midresp_struct(sIdx,iRewsize).midresp_peak_ix = peak_ix(peak_ix > 5 & peak_ix < 35) * bin_tbin;
        
        figure()
        imagesc(flipud(pValue_peth(driscoll_sort,:))) 
        colorbar()  
        title("p-value Heatmap")
        
        figure()
        imagesc(flipud(C(driscoll_sort,:))) 
        colorbar()  
        title("Convolution Heatmap")

%         figure()
%         imagesc(zscore(unshuffled_peth(peak_ix > 5 & peak_ix < 35,:),[],2)) 
%         caxis([-3,3]) 
    end 
    
%     for odd_even = [1,2] 
%         trial_selection = 1:odd_even:nTrials;
%         % Only select 4 uL trials for increased consistency
%         fr_mat = FR_decVar(sIdx).fr_mat(trial_selection);
%         fr_mat = cat(2,fr_mat{:}); 
%         timesince = FR_decVar(sIdx).decVarTimeSinceRew(trial_selection);
%         timesince = cat(2,timesince{:});   
%         nNeurons = size(fr_mat,1);
% 
%         % Bin activity and take mean per bin
%         [~,~,bin] = histcounts(timesince,decVar_bins);
%         unshuffled_peth = nan(nNeurons,max(bin));
%         for i = 1:max(bin)
%             unshuffled_peth(:,i) = mean(fr_mat(:,bin==i),2);
%         end
% 
%         % Find shuffle distribution 
%         nShuffles = 1000; 
%         shuffle_greater = zeros(size(fr_mat,1),max(bin));
%         for shuffle = 1:nShuffles 
%             fr_shuffle = fr_mat;
%             shifts = randi(size(fr_shuffle,2),nNeurons,1);
% 
%             parfor neuron = 1:nNeurons
%                 fr_shuffle(neuron,:) = circshift(fr_shuffle(neuron,:),shifts(neuron));
%             end
% 
%             i_shufflePETH = nan(nNeurons,max(bin));
%             for i = 1:max(bin)
%                 i_shufflePETH(:,i) = mean(fr_shuffle(:,bin==i),2);
%             end 
% 
%             [r,c] = find(i_shufflePETH > unshuffled_peth);  
%             ind = sub2ind(size(i_shufflePETH),r,c);
%             shuffle_greater(ind) = shuffle_greater(ind) + 1;
% 
%             if mod(shuffle,100) == 0 
%                 disp(shuffle)
%             end
%         end 
% 
%         % now find p value per bin per neuron
%         pValue_peth = shuffle_greater / nShuffles; 
%         pValue_peth = flipud(pValue_peth(neuron_order,:)); 
%         unshuffled_peth = flipud(unshuffled_peth(neuron_order,:));
% 
% %         figure()
% %         imagesc(flipud(pValue_peth(neuron_order,:))) 
% %         colorbar() 
%         
%         % find significant peaks
%         [r,c] = find(pValue_peth < .05); 
%         ind = sub2ind(size(pValue_peth),r,c); 
%         significant_peaks = zeros(size(pValue_peth)); 
%         significant_peaks(ind) = 1;   
% 
%         % perform convolution to find where we have structure (not just sig ix)
%         peak_width = 3;
%         C = conv2(significant_peaks,ones(1,peak_width),'same');
%         [r,c] = find(C >= 3); % find where we had significant in a row
%         peak_ix = nan(nNeurons,1);
%         for neuron = 1:nNeurons
%             peak_ix(neuron) = median(c(r == neuron));
%         end  
% 
%         figure()
%         imagesc(zscore(unshuffled_peth(peak_ix > 5 & peak_ix < 35,:),[],2)) 
%         caxis([-3,3]) 
%         % throw in after reward sizes
%         peak_ix_cell{sIdx,odd_even+4} = peak_ix;  
%         driscoll_midresp_struct(sIdx,odd_even+4).peak_ix = peak_ix * bin_tbin;
%         driscoll_midresp_struct(sIdx,odd_even+4).midresp = find(peak_ix > 5 & peak_ix < 35);
%     end
end 

%% Visualize results 
close all
for sIdx = 24 
%     session = sessions{sIdx}; 
%     data = load(fullfile(paths.data,session));   
%     rewsize = mod(data.patches(:,2),10);   
    nNeurons = length(index_sort_all{sIdx});
    figure();hold on
    for iRewsize = [1,2,4]
        scatter(1:nNeurons,driscoll_midresp_struct(sIdx,iRewsize).peak_ix,'.')
    end 
    legend("1 uL","2 uL","4 uL") 
    bins = -1.5:.05:1.5; 
    
    figure();hold on 
    histogram(driscoll_midresp_struct(sIdx,2).peak_ix - driscoll_midresp_struct(sIdx,1).peak_ix ,bins)
    histogram(driscoll_midresp_struct(sIdx,4).peak_ix - driscoll_midresp_struct(sIdx,2).peak_ix ,bins)
    histogram(driscoll_midresp_struct(sIdx,4).peak_ix - driscoll_midresp_struct(sIdx,1).peak_ix,bins) 
    legend("2 uL Peak - 1 uL Peak","4 uL Peak - 2 uL Peak","4 uL Peak - 1 uL Peak")  
    title("Distribution of peak location differences on Different Rewsize Trials") 
    xlabel("Peak time difference") 
    xline(0,'--','linewidth',2)
    
    figure()
    histogram(driscoll_midresp_struct(sIdx,6).peak_ix - driscoll_midresp_struct(sIdx,5).peak_ix) 
    title("Distribution of peak location differences on Odd/Even Trials") 
    xlabel("Peak time difference")
end