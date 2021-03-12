function [beta_shuffle,coeff,score,expl] = shuffle_coeffs(beta_all_sig,shuffle_type)
% Shuffle GLM coefficents and return results of PCA on shuffled matrix

    [n_vars,n_cells] = size(beta_all_sig); 
    beta_shuffle = nan(size(beta_all_sig));  
    % change perm matrix to modulate which coefficients get shuffled together 
    if shuffle_type == "complete" % permute all coefficients independently  
        perm_matrix = arrayfun(@(i_var) randperm(n_cells)',1:n_vars,'un',0); 
    elseif shuffle_type == "kern_together" % Keep reward kernels togther 
        perm_matrix = arrayfun(@(i_rewsize) [repmat(randperm(n_cells)',[1 11]) randperm(n_cells)' randperm(n_cells)' randperm(n_cells)'],1:3,'un',0);
    elseif shuffle_type == "taskvar_together"  % Keep coeffs together across reward sizes
        perm_one_rewsize = [repmat(randperm(n_cells)',[1 11]) randperm(n_cells)' randperm(n_cells)' randperm(n_cells)'];
        perm_matrix = arrayfun(@(i_rewsize) perm_one_rewsize, 1:3,'un',0); 
    else 
        throw(MException('MyComponent:noSuchVariable',"Shuffle type must be complete, kern_together, or taskvar_together"))
    end

    perm_matrix = cat(2,perm_matrix{:});
    for i_var = 1:n_vars 
        % to get guys shuffling together
        beta_shuffle(i_var,:) = beta_all_sig(i_var,perm_matrix(:,i_var));
    end

    % Shuffle reward kernel ix together, otherwise coeffs shuffle across cells
    beta_shuffle_norm = zscore(beta_shuffle);

    % beta_norm = beta_all_sig;
    [coeff,score,~,~,expl] = pca(beta_shuffle_norm');

end

