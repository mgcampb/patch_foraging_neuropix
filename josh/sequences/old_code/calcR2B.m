function [mean_r2b,r2b] = calcR2B(PETH,ridgeWidth,norm)
    % calculate ridge to background ratio for normalized responsivity PETH
    
    [nNeurons,nBins] = size(PETH);
    [~,peak_ix] = max(PETH,[],2); 
    r2b = zeros(nNeurons,1);
    for neuron = 1:nNeurons  
        ridgeStart = max(1,peak_ix(neuron)-ridgeWidth); 
        ridgeEnd = min(nBins-1,(peak_ix(neuron)+ridgeWidth));
        background =  mean(PETH(neuron,[1:ridgeStart,ridgeEnd:nBins]));
        ridge = mean(PETH(neuron,ridgeStart:ridgeEnd)); 
        if norm == "zscore"
            r2b(neuron) = (ridge + 3) / (background + 3); % add 3 to make all positive  
        elseif norm == "peak" 
            r2b(neuron) = ridge / background;
        end
    end 
    mean_r2b = nanmean(r2b);
end

