%% Try out a UMAP-> HDBSCAN first on digits dataset 

%% Load digits data 
oldpath = addpath(fullfile(matlabroot,'examples','nnet','main'));
mnist_directory = 'C:\Users\joshs\Documents\mnist_data';
filenameImagesTrain = 'train-images-idx3-ubyte.gz';
filenameLabelsTrain = 'train-labels-idx1-ubyte.gz';
filenameImagesTest = 't10k-images-idx3-ubyte.gz';
filenameLabelsTest = 't10k-labels-idx1-ubyte.gz';

XTrain = extractdata(processImagesMNIST(fullfile(mnist_directory,filenameImagesTrain)));
YTrain = processLabelsMNIST(fullfile(mnist_directory,filenameLabelsTrain));
XTest = extractdata(processImagesMNIST(fullfile(mnist_directory,filenameImagesTest)));
YTest = processLabelsMNIST(fullfile(mnist_directory,filenameLabelsTest));

XTrain_flat = reshape(XTrain,[28*28,60000]); 
XTest_flat = reshape(XTest,[28*28,10000]); 

%% Run UMAP 

min_dist_test = [.1 .15 .2 .3]; 
n_neighbors_test = [10 20 30 50]; 

XTrain_norm = zscore(XTrain_flat'); 

n_components = 2;

figure()
for i_n_neighbors = 1:numel(n_neighbors_test)
    n_neighbors = n_neighbors_test(i_n_neighbors);
    for i_min_dist = 1:numel(min_dist_test)
        min_dist = min_dist_test(i_min_dist);
        X_umap = run_umap(XTrain_norm,'n_components',n_components,'n_neighbors',n_neighbors,'min_dist',min_dist, 'verbose','none');
        subplot(numel(min_dist_test),numel(n_neighbors_test),i_n_neighbors + (i_min_dist - 1) * numel(n_neighbors_test))
%         scatter(X_umap(:,1),X_umap(:,2),3,'o')
        gscatter(X_umap(:,1),X_umap(:,2),YTrain,[],'o',3)
    end
end