function [SI,SInorm,log_r2b,entropy] = calculateSI(PETH,ridgeWidth,opt)
    % calculate sequentiality index ala Orhan and Ma 2019
    % SI = entropy of peak time distn + mean log ridge to background ratio 
    % returns SI and SInorm = (SI_raw - SI_ramp) / (SI_seq - SI_ramp)
    
    %%%% Read in kwargs %%%%
    norm = "zscore";
    if exist('opt', 'var') && isfield(opt,'norm')
        norm = opt.norm;
    end

    suppressVis = true;
    if exist('opt', 'var') && isfield(opt,'suppressVis')
        suppressVis = opt.suppressVis;
    end
    
    innerCall = false;
    if exist('opt', 'var') && isfield(opt,'innerCall')
        innerCall = opt.innerCall;
    end
    
    % some useful quantities
    [nNeurons,nBins] = size(PETH); 
    [~,peak_ix] = max(PETH,[],2); % find the location of peak activity 
    
    % zscore and add 3 sds to get rid of negative values if we are zscoring
    if norm == "zscore" 
        PETH = zscore(PETH,[],2) + 3; 
    elseif norm == "prezscore" 
        PETH = PETH + 3; % we zscored before, but now need to add 3
    elseif norm == "peak" 
        PETH = PETH ./ max(PETH,[],2); 
    elseif innerCall == false
        disp("No normalization performed in calculateSI!")
    end

    % Calculate ridge to background ratio
    log_r2b = zeros(nNeurons,1);
    for neuron = 1:nNeurons 
        ridgeStart = max(1,peak_ix(neuron)-ridgeWidth); 
        ridgeEnd = min(nBins-1,(peak_ix(neuron)+ridgeWidth));
        background =  mean(PETH(neuron,[1:ridgeStart,ridgeEnd:nBins]));
        ridge = mean(PETH(neuron,ridgeStart:ridgeEnd));
        log_r2b(neuron) = log(ridge / background);
    end 
    
    % Calculate entropy of peak distn  
    n_entropy_bins = 20; % from paper
    [counts,~] = histcounts(peak_ix,n_entropy_bins);
    counts = counts + .1; % to prevent log(0) 
    probabilities = counts / sum(counts); 
    entropy = -sum(probabilities .* log(probabilities)); % calculate shannon entropy! 
    
    % raw SI calculation
    SI = entropy + 3 * mean(log_r2b); % upweight ridge to background ratio
    
    if innerCall == false
        % Now find SI for ramping/sequence populations to get normalized value 
        % Sequence of gaussian activations 
        time = linspace(0,10,nBins);
        peaks = linspace(0,10,nNeurons);
        ideal_seq = flipud(normpdf(repmat(time,[nNeurons,1])',peaks,ones(1,nNeurons))'); 
        % Ramping activity (exponential with varying tau) 
        tau_min = .5; 
        tau_scaling = .1;
        taus = tau_min + tau_scaling * rand(nNeurons,1);
        ideal_ramp = exp(time .* taus);  
        % calculate baseline SIs
        opt_rec = opt; 
        opt_rec.innerCall = true; % break recursion 
        opt_rec.suppressVis = true; 
        opt_rec.norm = "zscore";
        SI_seq = calculateSI(ideal_seq,ridgeWidth,opt_rec);
        SI_ramp = calculateSI(ideal_ramp,ridgeWidth,opt_rec);  
        
        % calculate SInorm 
        SInorm = (SI - SI_ramp) / (SI_seq - SI_ramp);
    end
    
    if suppressVis == false 
        figure() 
        subplot(1,2,1)
        histogram(log_r2b)  
        title("Distribution of log ridge to background ratio")
        subplot(1,2,2) 
        histogram(peak_ix,n_entropy_bins)
        title("Distribution of peak counts") 
    end
    
end

