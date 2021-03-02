function [X_train,X_test,y_train,y_test] = kfold_split_singleSession(X_dataset,y_dataset,foldid,kFold) 
    % Concatenate features
    X_train_cell = X_dataset(foldid ~= kFold);
    X_train = cat(2,X_train_cell{:})';
    X_test_cell = X_dataset(foldid == kFold);
    X_test = cat(2,X_test_cell{:})';

    % Concatenate labels
    y_train_cell = y_dataset(foldid ~= kFold);
    y_train = cat(2,y_train_cell{:})';
    y_test_cell = y_dataset(foldid == kFold);
    y_test = cat(2,y_test_cell{:})';
end 


