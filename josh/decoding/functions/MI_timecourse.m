function [MI_timeBinned,H_given_hat,H_given_true,PMI_mat,MI_mat,CH_mat] = MI_timecourse(confusion_counts,stride)
%   Estimate timecourse of decoding fidelity using information theory metrics
    
    n_bins = size(confusion_counts,1); % (this is a square matrix)
%     % calculate mutual information per bin
%     for bin = 1:(n_bins-stride)
%         confusion_counts(
%     end 
    
    % joint probability
    p_xy = confusion_counts ./ sum(confusion_counts,'all'); %  
    % marginal probabilities
    p_x = sum(confusion_counts,1) ./ sum(confusion_counts,'all'); % p(y_hat = t)
    p_y = sum(confusion_counts,2) ./ sum(confusion_counts,'all'); % p(y_true = t)
    % probability distns over x and y 
%     p_y
    
    % iterate over and get contribution per timestep to mutual information
    PMI_mat = zeros(n_bins,n_bins); 
    MI_mat = zeros(n_bins,n_bins); 
    CH_mat = zeros(n_bins,n_bins); 
    for iy = 1:n_bins
        for ix = 1:n_bins
            if p_xy(iy,ix) > 0 && p_y(iy) > 0 && p_x(ix) > 0
                MI_mat(iy,ix) = p_xy(iy,ix) * log(p_xy(iy,ix)/(p_y(iy)*p_x(ix)));
                PMI_mat(iy,ix) = log(p_xy(iy,ix)/(p_y(iy)*p_x(ix))); % p_xy(iy,ix) * 
                CH_mat(iy,ix) = - p_xy(iy,ix) * log(p_xy(iy,ix)/p_x(ix));
            end
        end
    end 
    
    % now calculate MI per timebin
    MI_timeBinned = nan((n_bins - stride),1); 
    for i_bin = 1:(n_bins - stride) 
%         MI_timeBinned(i_bin) = sum(MI_mat(i_bin:i_bin+stride,i_bin:i_bin+stride),'all');
        MI_timeBinned(i_bin) = MI_confusionmat(confusion_counts(:,i_bin:i_bin+stride));
    end
    
    H_given_hat = NaN; 
    H_given_true = NaN; 
    
    
    
end

