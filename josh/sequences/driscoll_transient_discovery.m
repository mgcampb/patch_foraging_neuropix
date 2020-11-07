function transients_struct = driscoll_transient_discovery(FR_decVar,trial_selection,decVar_bins,tbin_ms,opt)
    % Find significant transients w.r.t. shuffle controls
    % Given a subset of trials to analyze (ie odd/even, rewsize, rew # 1)
    % Output struct

    %%%% Read in kwargs %%%%
    preRew_buffer = NaN;
    if exist('opt', 'var') && isfield(opt,'preRew_buffer')
        preRew_buffer = opt.preRew_buffer;
    end 

    postStop_buffer = NaN;
    if exist('opt', 'var') && isfield(opt,'postStop_buffer')
        postStop_buffer = opt.postStop_buffer;
    end   

    visualization = false; 
    if exist('opt', 'var') && isfield(opt,'visualization')
        visualization = opt.visualization;
    end    

    nShuffles = 1000; 
    if exist('opt','var') && isfield(opt,'nShuffles') 
        nShuffles = opt.nShuffles; 
    end

    %%%% Finished kwarg reading %%%% 

    % Only select one reward size at a time
    fr_mat = FR_decVar.fr_mat(trial_selection);
    fr_mat = cat(2,fr_mat{:});
    timesince = FR_decVar.decVarTimeSinceRew(trial_selection);
    timesince = cat(2,timesince{:});
    timepatch = FR_decVar.decVarTime(trial_selection);
    timepatch = cat(2,timepatch{:});
    nNeurons = size(fr_mat,1);

    % optionally take off some time before reward reception (smoothing)
    % would be nice to procedurally generate indices
    if ~isnan(preRew_buffer)
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
    if ~isnan(postStop_buffer)
        postStop_buffer = round(500 / tbin_ms);
        stop_ix = find(timepatch == tbin_ms / 1000);
        postStop_bool = false(length(timesince));
        for iTrial = 1:numel(stop_ix)
            postStop_bool(stop_ix(iTrial):stop_ix(iTrial) + postStop_buffer) = true;
        end
        fr_mat(:,postStop_bool) = [];
        timesince(postStop_bool) = [];
        %     timepatch(postStop_bool) = [];
    end

    % Bin activity and take mean per bin
    [~,~,bin] = histcounts(timesince,decVar_bins);
    unshuffled_peth = nan(nNeurons,max(bin));
    for i = 1:max(bin)
        unshuffled_peth(:,i) = mean(fr_mat(:,bin==i),2);
    end

    % Find shuffle distribution
    shuffle_greater = zeros(size(fr_mat,1),max(bin)); 
    bar = waitbar(0,"Creating Shuffle Distribution"); % progress tracking
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
        waitbar(shuffle/nShuffles,bar)
    end 
    close(bar)

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
    bin_tbin = decVar_bins(2) - decVar_bins(1); % convert bins to seconds
    transients_struct.peak_ix = peak_ix * bin_tbin;
    transients_struct.midresp = find(peak_ix > 5 & peak_ix < 35);
    transients_struct.midresp_peak_ix = peak_ix(peak_ix > 5 & peak_ix < 35) * bin_tbin;

    if visualization == true 
        % sort for quick and dirty visualization
        figure()
        imagesc(flipud(pValue_peth(driscoll_sort,:))) 
        colorbar()  
        title("p-value Heatmap")
        
        figure()
        imagesc(flipud(C(driscoll_sort,:))) 
        colorbar()  
        title("Convolution Heatmap")
        
        figure()
        imagesc(flipud(zscore(unshuffled_peth(transients_struct.midresp(midresp_peaksort),:),[],2)))
        caxis([-3,3]) 
        colorbar()  
        title("Z-Scored Mid-responsive Neuron Activity") 
    end
end

