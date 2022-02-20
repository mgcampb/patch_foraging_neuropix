function ari = gmm_ari_analysis(data,n_repeats,n_clusters,lambda,gmm_opt,options)
    % Perform bootstrapping to perform ARI analysis for cluster stability
    
    % Now perform rand index bootstrapping procedure
    [~ , bootsam_idx] = bootstrp(n_repeats,[],data);

    ari = nan(nchoosek(n_repeats,2),1);

    gmm_idx = nan(size(data,1),n_repeats);
    for i_repeat = 1:n_repeats
        try
            boot_gmm = fitgmdist(data(bootsam_idx(:,i_repeat),:),n_clusters,'RegularizationValue',lambda,'replicates',gmm_opt.replicates,'options',options);
            this_gmm_idx = cluster(boot_gmm,data); % cluster original data
            if length(unique(this_gmm_idx)) == n_clusters
                gmm_idx(:,i_repeat) = this_gmm_idx; % only allow clustering if has actually assigned to n_clusters
            else
                gmm_idx(:,i_repeat) = nan(size(data,1),1);
            end
        catch
            gmm_idx(:,i_repeat) = nan(size(data,1),1); % cluster original data
        end
    end
    
    % compute ARI over unique pairs
    counter = 1;
    for i = 1:n_repeats
        for j = 1:n_repeats
            if j < i
                if all(gmm_idx(:,i) == gmm_idx(:,j))
                    ari(counter) = 1;
                elseif any(isnan([gmm_idx(:,i) gmm_idx(:,j)]))
                    ari(counter) = nan; % one of these repeats didnt converge!
                else
                    ari(counter) = rand_index(gmm_idx(:,i),gmm_idx(:,j),'adjusted','fancy');
                end
                counter = counter + 1;
            end
        end
    end
end

