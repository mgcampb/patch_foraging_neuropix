function [X_train,X_test,y_train,y_test] = kfold_split(X_dataset,y_dataset,foldid,kFold,sessionIx) 
% kfold_split splits a dataset (cell array) into train and test folds 
%   X (features array) is concatenated as blocks by session and padded to allow for missing values
%   y (labels array) is concatenated and returned as a column vector 

    % Concatenate and pad features  
    X_train = padCat(X_dataset(foldid ~= kFold),sessionIx(foldid ~= kFold)); 
    X_test = padCat(X_dataset(foldid == kFold),sessionIx(foldid == kFold));    
    
    % Concatenate labels
    y_train_cell = y_dataset(foldid ~= kFold);
    y_train = cat(2,y_train_cell{:})';
    y_test_cell = y_dataset(foldid == kFold);
    y_test = cat(2,y_test_cell{:})';
end