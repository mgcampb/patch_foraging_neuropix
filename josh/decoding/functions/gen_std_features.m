function these_features = gen_std_features(clusters_session)
%GEN_STD_FEATURES Standard feature generation function for Naive Bayes Decoding
% 1:4 - clusters 1:4 
% 5: all GLM neurons 
% 6: Velocity 
% 7: Position
    these_features = cell(7,1); 
    for iCluster = 1:4
        if ~isempty(find(clusters_session == iCluster,1))
            these_features{iCluster} = struct;
            these_features{iCluster}.type = "KMeans Clusters";
            these_features{iCluster}.ix = iCluster; % indices within the feature type we selected
            these_features{iCluster}.shuffle = false; % shuffle?
            these_features{iCluster}.name = sprintf("Cluster %i",iCluster); % name for visualizations 
        end
    end 
    % add all clusters to end of the features
    these_features{5} = struct;
    these_features{5}.type = "KMeans Clusters";
    these_features{5}.ix = [1 2 3 4]; % indices within the feature type we selected
    these_features{5}.shuffle = false; % shuffle?
    these_features{5}.name = "All Clusters"; % name for visualizations 
    % Velocity
    these_features{6} = struct;
    these_features{6}.type = "Velocity";
    these_features{6}.ix = []; % indices within the feature type we selected
    these_features{6}.shuffle = false; % shuffle?
    these_features{6}.name = "Velocity"; % name for visualizations
    % Position
    these_features{7} = struct;
    these_features{7}.type = "Position";
    these_features{7}.ix = []; % indices within the feature type we selected
    these_features{7}.shuffle = false; % shuffle?
    these_features{7}.name = "Position"; % name for visualizations
end

