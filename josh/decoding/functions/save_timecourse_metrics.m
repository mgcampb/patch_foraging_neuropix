function timecourse_results = save_timecourse_metrics(cells_picked,nNeurons,timecourse_counter,dataset_opt,this_X,this_y_true,foldID,these_models)
    % Analyze timecourse of decoding fidelity at certain point in one by
    % one decoding
    
    nan_cells = setdiff(1:nNeurons,cells_picked); % cells to set to nan
    % Predict with the new
    i_y_hat_tmp = cell(numel(dataset_opt.numFolds),1);
    for kFold = 1:dataset_opt.numFolds
        X_test = this_X(foldID == kFold);
        X_test = cat(2,X_test{:})';
        X_test(:,nan_cells) = NaN;

        i_y_hat = predict(these_models{kFold},X_test);

        i_y_hat_tmp{kFold} = i_y_hat;
    end
    i_y_hat_full = cat(1,i_y_hat_tmp{:})';

    confusionmat_this = confusionmat(this_y_true,i_y_hat_full);

    abs_errors = abs(this_y_true - i_y_hat_full);
    sq_errors = (this_y_true - i_y_hat_full).^2;
    mae_mean_timecourse = nan(max(this_y_true),1);
    mae_sem_timecourse = nan(max(this_y_true),1); 
    rmse_mean_timecourse = nan(max(this_y_true),1);
    rmse_sem_timecourse = nan(max(this_y_true),1); 
    yhat_mean_timecourse = nan(max(this_y_true),1);
    yhat_sem_timecourse = nan(max(this_y_true),1);
    for true_time = 1:max(this_y_true)
        mae_mean_timecourse(true_time) = nanmean(abs_errors(this_y_true == true_time));
        mae_sem_timecourse(true_time) = 1.96 * nanstd(abs_errors(this_y_true == true_time)) / length(find(this_y_true == true_time)); 
        rmse_mean_timecourse(true_time) = sqrt(nanmean(sq_errors(this_y_true == true_time)));
        rmse_sem_timecourse(true_time) = 1.96 * nanstd(sq_errors(this_y_true == true_time)) / length(find(this_y_true == true_time)); 
        yhat_mean_timecourse(true_time) = nanmean(i_y_hat_full(this_y_true == true_time));
        yhat_sem_timecourse(true_time) = 1.96 * nanstd(i_y_hat_full(this_y_true == true_time)) / length(find(this_y_true == true_time));
    end

    % log results
    timecourse_results = struct;
    timecourse_results.y_hat = i_y_hat_full;
    if timecourse_counter == 1 % save y_true if this is the first savepoint
        timecourse_results.y_true = this_y_true;
    end
    timecourse_results.confusionmat = confusionmat_this;
    timecourse_results.mae_mean_timecourse = mae_mean_timecourse;
    timecourse_results.mae_sem_timecourse = mae_sem_timecourse; 
    timecourse_results.rmse_mean_timecourse = rmse_mean_timecourse;
    timecourse_results.rmse_sem_timecourse = rmse_sem_timecourse; 
    timecourse_results.yhat_mean_timecourse = yhat_mean_timecourse;
    timecourse_results.yhat_sem_timecourse = yhat_sem_timecourse;
end

