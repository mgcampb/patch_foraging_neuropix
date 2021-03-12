function entropy = calc_shannonH(X)
%CALC_SHANNONH Calculate Shannon Entropy of discrete r.v. X

    counts = histcounts(X); % get probability per value of X
    probabilities = counts / sum(counts); 
    entropy = -sum(probabilities(probabilities ~= 0) .* log(probabilities(probabilities ~= 0))); % calculate shannon entropy! 

end

