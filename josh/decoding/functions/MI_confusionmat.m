function mutual_information = MI_confusionmat(confusionCounts)
% Calculate Mutual information between decoded and true variable given a
% confusion chart w/ counts 
    
    % joint probability
    p_xy = confusionCounts ./ sum(confusionCounts,'all');  
    % marginal probabilities
    p_x = sum(confusionCounts,1) ./ sum(confusionCounts,'all');
    p_y = sum(confusionCounts,2) ./ sum(confusionCounts,'all'); 
    
    mutual_information = 0; 
    for iy = 1:size(confusionCounts,1)
        for ix = 1:size(confusionCounts,2) 
            if p_xy(iy,ix) > 0 && p_y(iy) > 0 && p_x(ix) > 0
                mutual_information = mutual_information + p_xy(iy,ix) * log(p_xy(iy,ix)/(p_y(iy)*p_x(ix)));
            end
        end
    end
end

