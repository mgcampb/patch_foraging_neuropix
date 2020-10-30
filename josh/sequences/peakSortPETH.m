function [sorted_peth,neuron_order,unsorted_peth_norm] = peakSortPETH(FR_decVar,dvar,decVar_bins,opt)
% Sort neurons by peak firing rate bin according to time or time since last
% reward
    
    %%%% Read in kwargs %%%%
    norm = "zscore";
    if exist('opt', 'var') && isfield(opt,'norm')
        norm = opt.norm;
    end

    suppressVis = false;
    if exist('opt', 'var') && isfield(opt,'suppressVis')
        suppressVis = opt.suppressVis;
    end
    
    shuffle = false;
    if exist('opt', 'var') && isfield(opt,'shuffle')
        shuffle = opt.shuffle;
    end
   
    nTrials = length(FR_decVar.fr_mat);
    trials = 1:nTrials;
    if exist('opt', 'var') && isfield(opt,'trials')
        if ~strcmp(opt.trials,'all')
            trials = opt.trials;
        end
    end 
    
    neurons = 'all';
    if exist('opt','var') && isfield(opt,'neurons')  
        if ~strcmp(opt.neurons,'all') 
            neurons = opt.neurons;  
        end 
    end
    
    %%%% set alignment variable %%%%
    if dvar == "time"
        decVar_cell = FR_decVar.decVarTime;
        label = "Time on Patch";
    elseif dvar == "timesince"
        decVar_cell = FR_decVar.decVarTimeSinceRew;
        label = "Time Since Last Reward";
    else
        disp("Please input time or timesince as alignment variable") 
    end

    %%%% prep decision variable bins %%%%
    if length(trials) == nTrials
        fr_mat = cat(2,FR_decVar.fr_mat{:});
        decVar = cat(2,decVar_cell{:});
    else  % subselect trials
        frCell = FR_decVar.fr_mat(trials);
        fr_mat = cat(2,frCell{:});
        decVarCell = decVar_cell(trials);
        decVar = cat(2,decVarCell{:});
    end 
    
    % shuffle by random rotation if told to
    shifts = randi(size(fr_mat,2),size(fr_mat,1),1);
    if shuffle == true
        parfor neuron = 1:size(fr_mat,1)
            fr_mat(neuron,:) = circshift(fr_mat(neuron,:),shifts(neuron));
        end
    end

    % perform averaging over bins step with histcounts
    [~,~,bin] = histcounts(decVar,decVar_bins);
    unsorted_peth = nan(size(fr_mat,1),max(bin));
    for i = 1:max(bin)
        unsorted_peth(:,i) = mean(fr_mat(:,bin==i),2);
    end

    if norm == "zscore"
        unsorted_peth_norm = zscore(unsorted_peth,[],2);
    elseif norm == "peak"
        unsorted_peth_norm = unsorted_peth ./ max(unsorted_peth,[],2);
        unsorted_peth_norm(isnan(unsorted_peth_norm)) = 0; 
    elseif norm == "none" 
        unsorted_peth_norm = unsorted_peth;
    end
    
    [~,index] = max(unsorted_peth'); 
    
    [~,neuron_order] = sort(index); 
    sorted_peth = unsorted_peth_norm(neuron_order,:); 
    
    % subselect neurons? 
    if ~strcmp(neurons,'all') 
        sorted_peth = sorted_peth(neurons,:);
    end
    
    % now making xticks at even seconds
    max_round = floor(max(decVar_bins));
    secs = 0:max_round;
    x_idx = [];
    for i = secs
        x_idx = [x_idx find(decVar_bins > i,1)];
    end
    
    if suppressVis == false
        figure()
        colormap('jet')
        imagesc(flipud(sorted_peth));
        colorbar()
        colormap('jet')
        xlim([1,numel(decVar_bins)-1])
        if shuffle == false
            xlabel([label; " (ms)"])
            title(sprintf("PETH Sorted to %s",label))
        else
            xlabel([label; " Shuffled  (ms)"])
            title(sprintf("PETH Sorted to Shuffled %s",label))
        end
        xticks(x_idx)
        xticklabels(secs * 1000)
        ylabel("Neurons")
    end
end

