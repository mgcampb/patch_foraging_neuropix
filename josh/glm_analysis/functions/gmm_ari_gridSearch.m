function ari = gmm_ari_gridSearch(data,n_repeats,lambdas,n_clusters,gmm_opt,options)

    % Perform bootstrapping to perform ARI analysis for cluster stability
    % over choices of n_clusters and lambda for fitting a Gaussian mixture
    % model
    
    % Now perform rand index bootstrapping procedure
    [~ , bootsam_idx] = bootstrp(n_repeats,[],data);

    ari = nan(numel(lambdas),numel(n_clusters),nchoosek(n_repeats,2));

    b = waitbar(0,'Performing ARI analysis');
    compute_counter = 0;
    for i_lambda = 1:numel(lambdas)
        this_lambda = lambdas(i_lambda);

        for i_n_clusters = 1:numel(n_clusters)
            this_n_clusters = n_clusters(i_n_clusters);

            gmm_idx = nan(size(data,1),n_repeats);
            for i_repeat = 1:n_repeats
                boot_gmm = fitgmdist(data(bootsam_idx(:,i_repeat),:),this_n_clusters,'RegularizationValue',this_lambda,'replicates',gmm_opt.replicates,'options',options);
                gmm_idx(:,i_repeat) = cluster(boot_gmm,data); % cluster original data
            end

            % compute ARI over unique pairs
            counter = 1;
            for i = 1:n_repeats
                for j = 1:n_repeats
                    if j < i
                        if all(gmm_idx(:,i) == gmm_idx(:,j))
                            ari(i_lambda,i_n_clusters,counter) = 1;
                        else
                            ari(i_lambda,i_n_clusters,counter) = rand_index(gmm_idx(:,i),gmm_idx(:,j),'adjusted','fancy');
                        end
                        counter = counter + 1;
                    end
                end
            end

            compute_counter = compute_counter + 1;
            waitbar(compute_counter / (numel(lambdas) * numel(n_clusters)));

        end
    end
    close(b);
end

