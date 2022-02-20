%% Try out a UMAP-> HDBSCAN first on digits dataset 

%% Load digits data 
oldpath = addpath(fullfile(matlabroot,'examples','nnet','main'));
mnist_directory = '/Users/joshstern/Documents/mnist_dataset';
filenameImagesTrain = 'train-images-idx3-ubyte';
filenameLabelsTrain = 'train-labels-idx1-ubyte';
filenameImagesTest = 't10k-images-idx3-ubyte';
filenameLabelsTest = 't10k-labels-idx1-ubyte';

XTrain = processImagesMNIST(fullfile(mnist_directory,filenameImagesTrain));

YTrain = processLabelsMNIST(fullfile(mnist_directory,filenameLabelsTrain));
XTest = processImagesMNIST(fullfile(mnist_directory,filenameImagesTest));
YTest = processLabelsMNIST(fullfile(mnist_directory,filenameLabelsTest));

function X = read_mnist_image