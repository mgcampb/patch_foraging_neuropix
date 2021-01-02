function [transients_struct,unshuffled_peths,pvalue_peths] = driscoll_transient_discovery2(fr_mat_trials,task_vars_trialed,trial_selection,tbin_ms,var_bins,opt)
% DRISCOLL_TRANSIENT_DISCOVERY2 
% Electric boogaloo. 
% Updated taskvar transient discovery within new framework that allows cue
% transients as well.   
% Arguments: fr_mat_trials (one session), task_vars_trialed (one session),
%            trial_selection
    
    %%%% Read in kwargs %%%%
    preRew_buffer = NaN;
    if exist('opt', 'var') && isfield(opt,'preRew_buffer')
        preRew_buffer = opt.preRew_buffer;
    end 

    visualization = false; 
    if exist('opt', 'var') && isfield(opt,'visualization')
        visualization = opt.visualization;
    end    

    nShuffles = 1000; 
    if exist('opt','var') && isfield(opt,'nShuffles') 
        nShuffles = opt.nShuffles; 
    end  
    
    sig_threshold = .01; 
    if exist('opt','var') && isfield(opt,'sig_threshold') 
        sig_threshold = opt.sig_threshold; 
    end 
    
    consec_peak_width = 3; 
    if exist('opt','var') && isfield(opt,'peak_width') 
        consec_peak_width = opt.consec_peak_width; 
    end 
    
    % find selectivity for some of the variables 
    vars = 1:3; 
    if exist('opt','var') && isfield(opt,'vars') 
        vars = opt.vars; 
    end

    %%%% Finished kwarg reading %%%% 
    fr_mat = fr_mat_trials(trial_selection); 
    fr_mat = cat(2,fr_mat{:}); 
    task_vars = task_vars_trialed(trial_selection); 
    task_vars = cat(2,task_vars{:}); 

    nNeurons = size(fr_mat,1);

    % optionally take off some time before reward reception (smoothing)
    if ~isnan(preRew_buffer)
        % Remove data before rewards at t = 0 (for cue selectivity analysis)
        rew0_ix = find(task_vars(2,:) == tbin_ms / 1000);
        preRew0_bool = false(length(task_vars));
        for iRew = 1:numel(rew0_ix)
            preRew0_bool(rew0_ix(iRew) - preRew_buffer:rew0_ix(iRew)) = true;
        end
        fr_mat(:,preRew0_bool) = [];
        task_vars(:,preRew0_bool) = [];
        
        % Remove data before rewards at t = 1+
        rew1plus_ix = find(task_vars(3,:) == tbin_ms / 1000);
        preRew1_bool = false(length(task_vars));
        for iRew = 1:numel(rew1plus_ix)
            preRew1_bool(rew1plus_ix(iRew) - preRew_buffer:rew1plus_ix(iRew)) = true;
        end
        fr_mat(:,preRew1_bool) = [];
        task_vars(:,preRew1_bool) = [];
    end 
     
    transients_struct = struct;  
    unshuffled_peths = cell(numel(vars),1); 
    pvalue_peths = cell(numel(vars),2);  
    counter = 1; 
    for iVar = vars 
        this_task_var = task_vars(iVar,:);
        % Bin activity and take mean per bin
        [~,~,bin] = histcounts(this_task_var,var_bins{iVar});
        unshuffled_peth = nan(nNeurons,length(var_bins{iVar})-1);
        for i = 1:max(bin)
            unshuffled_peth(:,i) = mean(fr_mat(:,bin==i),2);
        end 
        unshuffled_peths{iVar} = unshuffled_peth;

        % Find shuffle distribution by shifting stimulus, paying attention
        % to where it was NaN beforehand
        shuffle_greater = zeros(size(fr_mat,1),length(var_bins{iVar})-1);  
        shuffle_lesser = zeros(size(fr_mat,1),length(var_bins{iVar})-1);  
%         bar = waitbar(0,"Creating Shuffle Distribution"); % progress tracking 
        non_nan_ix = ~isnan(this_task_var);
        for shuffle = 1:nShuffles
            % shuffle task variable by rotational shift 
            this_task_var_shuffle = this_task_var;
            
            shift = randi(length(this_task_var_shuffle(non_nan_ix)),1);  
            this_task_var_shuffle(non_nan_ix) = circshift(this_task_var_shuffle(non_nan_ix),shift); 

            % Bin activity and take mean per bin
            [~,~,bin_shuffle] = histcounts(this_task_var_shuffle,var_bins{iVar});
            i_shufflePETH = nan(nNeurons,length(var_bins{iVar})-1);
            for i = 1:max(bin_shuffle)
                i_shufflePETH(:,i) = mean(fr_mat(:,bin_shuffle==i),2);
            end

            % add 1 to the locations where the shuffled PETH was greater than unshuffled
            [r,c] = find(i_shufflePETH >= unshuffled_peth);
            ind = sub2ind(size(i_shufflePETH),r,c);
            shuffle_greater(ind) = shuffle_greater(ind) + 1; 
            
            % add 1 to the locations where the shuffled PETH was greater than unshuffled
            [r,c] = find(i_shufflePETH <= unshuffled_peth);
            ind = sub2ind(size(i_shufflePETH),r,c);
            shuffle_lesser(ind) = shuffle_lesser(ind) + 1;

            % track progress
%             waitbar(shuffle/nShuffles,bar)
        end
%         close(bar)

        % now find p value per bin per neuron
        pValue_peth_greater = shuffle_greater / nShuffles;
        pvalue_peths{iVar,1} = pValue_peth_greater;
        
        % find significant peaks
        [r,c] = find(pValue_peth_greater < sig_threshold);
        ind = sub2ind(size(pValue_peth_greater),r,c);
        significant_peaks = zeros(size(pValue_peth_greater));
        significant_peaks(ind) = 1;

        % perform convolution to denoise a bit
        C_greater = conv2(significant_peaks,ones(1,consec_peak_width),'same');
        % find peak indices
        peak_ix_pos = nan(nNeurons,1);
        doublepeak_pos = false(nNeurons,1);
        for neuron = 1:nNeurons
            neuron_sig_ix = find(C_greater(neuron,:) >= consec_peak_width); % find where we had significant peak in a row
            % handle double peak
            if ~isempty(find(neuron_sig_ix(2:end) - neuron_sig_ix(1:end-1) > 1,1))
                doublepeak_pos(neuron) = true;
            end
            [~,neuron_peak_ix] = max(unshuffled_peth(neuron,neuron_sig_ix));
            if ~isempty(neuron_peak_ix)
                peak_ix_pos(neuron) = neuron_sig_ix(neuron_peak_ix);
            end
        end

        % log information
        bin_tbin = diff(var_bins{iVar}(1:2)); % convert bins to seconds
        transients_struct(counter).peak_ix_pos = peak_ix_pos * bin_tbin; 
        transients_struct(counter).doublepeak_pos = doublepeak_pos; % double peak?   
        
        %%% Now repeat for significant dips in activity %%%
        
        % now find p value per bin per neuron
        pValue_peth_lesser = shuffle_lesser / nShuffles; 
        pvalue_peths{iVar,2} = pValue_peth_lesser;
        
        % find significant peaks
        [r,c] = find(pValue_peth_lesser < sig_threshold);
        ind = sub2ind(size(pValue_peth_lesser),r,c);
        significant_peaks = zeros(size(pValue_peth_lesser));
        significant_peaks(ind) = 1;

        % perform convolution to denoise a bit
        C_lesser = conv2(significant_peaks,ones(1,consec_peak_width),'same');
        % find peak indices
        peak_ix_neg = nan(nNeurons,1);
        doublepeak_neg = false(nNeurons,1);
        for neuron = 1:nNeurons
            neuron_sig_ix = find(C_lesser(neuron,:) >= consec_peak_width); % find where we had significant peak in a row
            % handle double peak
            if ~isempty(find(neuron_sig_ix(2:end) - neuron_sig_ix(1:end-1) > 1,1))
                doublepeak_neg(neuron) = true;
            end
            [~,neuron_peak_ix] = min(unshuffled_peth(neuron,neuron_sig_ix));
            if ~isempty(neuron_peak_ix)
                peak_ix_neg(neuron) = neuron_sig_ix(neuron_peak_ix);
            end
        end

        % log information
        transients_struct(counter).peak_ix_neg = peak_ix_neg * bin_tbin; 
        transients_struct(counter).doublepeak_neg = doublepeak_neg; % double peak?  
        
        if visualization == true  
            figure()
            % sort for visualization
            [~,driscoll_sort] = sort(peak_ix_pos);
            driscoll_sort = driscoll_sort(ismember(driscoll_sort,find(~isnan(peak_ix_pos)))); % get rid of non significant cells
            % sort for quick and dirty visualization
            subplot(2,3,1)
            imagesc(flipud(log(pValue_peth_greater(driscoll_sort,:))) )
            colorbar()  
            title("Positive p-value Heatmap") 
            subplot(2,3,2)
            imagesc(flipud(C_greater(driscoll_sort,:))) 
            colorbar()  
            title("Positive Convolution Heatmap")
            subplot(2,3,3)
            imagesc(flipud(zscore(unshuffled_peth(driscoll_sort,:),[],2)))
            caxis([-3,3]) 
            colorbar()
            title("Z-Scored Neuron Responsivity")   
            
            % same for negative 
            % sort for quick and dirty visualization
            subplot(2,3,4)
            imagesc(flipud(log(pValue_peth_lesser(driscoll_sort,:))) )
            colorbar()  
            title("Negative p-value Heatmap") 
            subplot(2,3,5)
            imagesc(flipud(C_lesser(driscoll_sort,:))) 
            colorbar()  
            title("Negative Convolution Heatmap")
            subplot(2,3,6)
            imagesc(flipud(zscore(unshuffled_peth(driscoll_sort,:),[],2)))
            caxis([-3,3]) 
            colorbar()
            title("Z-Scored Neuron Responsivity") 
        end 
        counter = counter + 1; 
    end
end

