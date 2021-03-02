function [accuracies,precisions,TP_rates,FP_rates,ROC_AUC,PR_AUC] = logReg_eval(X,y,thresholds,xval_opt)

%LOGREG_EVAL Summary of this function goes here
%   Detailed explanation goes here
% 
    accuracies = nan(xval_opt.numFolds,numel(thresholds));
    precisions = nan(xval_opt.numFolds,numel(thresholds));
    TP_rates = nan(xval_opt.numFolds,numel(thresholds));
    FP_rates = nan(xval_opt.numFolds,numel(thresholds));
    ROC_AUC = nan(xval_opt.numFolds,1);
    PR_AUC = nan(xval_opt.numFolds,1);
    for fIdx = 1:xval_opt.numFolds
        % separate training and test data 
        if size(X,1) == 1
            data_train = X(xval_opt.foldid~=fIdx);  
            data_test = X(xval_opt.foldid==fIdx); 
        else 
            data_train = X(xval_opt.data_ix,xval_opt.foldid~=fIdx);  
            data_test = X(xval_opt.data_ix,xval_opt.foldid==fIdx); 
        end
        labels_train = y(xval_opt.foldid~=fIdx);
        labels_test = y(xval_opt.foldid==fIdx);

        % now fit logistic regression to our training data
        [B_velocity,~,~] = mnrfit(data_train',labels_train);
        pi_test = mnrval(B_velocity,data_test');

        for tIdx = 1:numel(thresholds)
            threshold = thresholds(tIdx);
            model_labels = double(pi_test(:,2) > threshold);
            cm = confusionmat(labels_test' - 1,model_labels);
            TN = cm(1,1);
            FN = cm(2,1);
            TP = cm(2,2);
            FP = cm(1,2);

            % classification performance metrics
            accuracy = (TP + TN) / sum(cm(:));
            precision = TP / (TP + FP); % precision: P(Yhat = 1 | Y = 1)
            TP_rate = TP / (TP + FN); % sensitivity or recall:  P(Yhat = 1 | Y = 1)
            FP_rate = FP / (TN + FP); % 1 - sensitivity: P(Yhat = 1 | Y = 0)

            % log metrics
            accuracies(fIdx,tIdx) = accuracy;
            precisions(fIdx,tIdx) = precision;
            TP_rates(fIdx,tIdx) = TP_rate;
            FP_rates(fIdx,tIdx) = FP_rate;
        end

        ROC_AUC_dx_vel = -squeeze(diff(FP_rates(fIdx,:)));
        ROC_AUC(fIdx) = sum(ROC_AUC_dx_vel .* TP_rates(fIdx,1:end-1));
        PR_AUC_dx_vel = -squeeze(diff(TP_rates(fIdx,:)));
        PR_AUC(fIdx) = sum(PR_AUC_dx_vel(~isnan(precisions(fIdx,1:end-1))) .* precisions(fIdx,~isnan(precisions(fIdx,1:end-1))));
    end
end

