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

sessions = dir(fullfile(paths.data,'*.mat'));
sessions = {sessions.name}; 

%% Extract FR matrices and timing information 
FR_decVar = struct; 
FRandTimes = struct;
for sIdx = 24:24
    buffer = 500;
    [FR_decVar_tmp,~] = genSeqStructs(paths,sessions,opt,sIdx,buffer);
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
for sIdx = 24:24 
    session = sessions{sIdx}; 
    data = load(fullfile(paths.data,session));  
    rewsize = mod(data.patches(:,2),10); 
    trial_selection = find(rewsize == 4);
    
    % Only select 4 uL trials for increased consistency
    fr_mat = FR_decVar(sIdx).fr_mat(trial_selection);
    fr_mat = cat(2,fr_mat{:}); 
    timesince = FR_decVar(sIdx).decVarTimeSinceRew(trial_selection);
    timesince = cat(2,timesince{:});   
    nNeurons = size(fr_mat,1);
    
    % Bin activity and take mean per bin
    [~,~,bin] = histcounts(timesince,decVar_bins);
    unshuffled_peth = nan(nNeurons,max(bin));
    for i = 1:max(bin)
        unshuffled_peth(:,i) = mean(fr_mat(:,bin==i),2);
    end
    
    % Find shuffle distribution 
    nShuffles = 1000; 
    shuffle_greater = zeros(size(fr_mat,1),max(bin));
    for shuffle = 1:nShuffles 
        fr_shuffle = fr_mat;
        shifts = randi(size(fr_shuffle,2),nNeurons,1);
        
        parfor neuron = 1:nNeurons
            fr_shuffle(neuron,:) = circshift(fr_shuffle(neuron,:),shifts(neuron));
        end
        
        i_shufflePETH = nan(nNeurons,max(bin));
        for i = 1:max(bin)
            i_shufflePETH(:,i) = mean(fr_shuffle(:,bin==i),2);
        end 
        
        [r,c] = find(i_shufflePETH > unshuffled_peth);  
        ind = sub2ind(size(i_shufflePETH),r,c);
        shuffle_greater(ind) = shuffle_greater(ind) + 1;
        
        % shuffle_peths(shuffle,:,:) = i_shufflePETH;  
        if mod(shuffle,100) == 0 
            disp(shuffle)
        end
    end 
    
    % now find p value per bin per neuron
    pValue_peth = shuffle_greater / nShuffles; 
    
    % Now visualize p values with sort
    opt.norm = "zscore";
    opt.trials = trial_selection;
    opt.suppressVis = true;
    dvar = "timesince";
    [sorted_peth,neuron_order,unsorted_peth] = peakSortPETH(FR_decVar(sIdx),dvar,decVar_bins,opt);
    index_sort_all{sIdx} = neuron_order;  
    
    pValue_peth = flipud(pValue_peth(neuron_order,:)); 
    unshuffled_peth = flipud(unshuffled_peth(neuron_order,:));
    
    figure()
    imagesc(flipud(pValue_peth)) 
    colorbar()
end 

%% Now find significant peaks by having low pValue for x consecutive indices
for sIdx = 24  
    % find significant peaks
    [r,c] = find(pValue_peth < .05); 
    ind = sub2ind(size(pValue_peth),r,c); 
    significant_peaks = zeros(size(pValue_peth)); 
    significant_peaks(ind) = 1;   
    
    % perform convolution to find where we have structure (not just sig ix)
    peak_width = 3;
    C = conv2(significant_peaks,ones(1,peak_width),'same');
    
    [r,c] = find(C >= 3);  
    peak_ix = nan(nNeurons,1);
    for neuron = 1:nNeurons
        peak_ix(neuron) = median(c(r == neuron));
    end  
    
    figure()
    imagesc(zscore(unshuffled_peth(peak_ix > 5 & peak_ix < 35,:),[],2)) 
    caxis([-3,3])
    
end
