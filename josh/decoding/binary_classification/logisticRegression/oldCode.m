%% Now perform classification with logistic regression, using k-fold x-val  
% add velocity classification as a control
close all 
figcounter = 1;
for sIdx = 1:numel(sessions)
    session = sessions{sIdx}(1:end-4); 
    session_title = sessions{sIdx}([1:2 end-6:end-4]);
    data = load(fullfile(paths.data,session)); 
    patches = data.patches;
    patchCSL = data.patchCSL;
    prts = patchCSL(:,3) - patchCSL(:,2);
    patchType = patches(:,2);
    rewsize = mod(patchType,10);  
    
    all_concat_PCs_noPreRew = horzcat(classification_struct(sIdx).PCs_noPreRew{:});   
    all_concat_labels_noPreRew = horzcat(classification_struct(sIdx).labels_noPreRew{:}) + 1;  
    all_concat_rewsize_noPrewRew = vertcat(classification_struct(sIdx).rewsize_noPreRew{:})'; 
    all_concat_vel_noPreRew = horzcat(classification_struct(sIdx).vel_noPreRew{:});  
%     all_concat_meanRamp_upAll_noPreRew = horzcat(classification_struct(sIdx).meanRamp_upAll_noPreRew{:});
%     all_concat_meanRamp_upCommon_noPreRew = horzcat(classification_struct(sIdx).meanRamp_upCommon_noPreRew{:});
    
    % folds are going to be over points that did not directly precede reward
    points = 1:numel(all_concat_rewsize_noPrewRew);
    
    % make folds
    xval_opt = struct;
    xval_opt.numFolds = 10;
    xval_opt.rew_size = [1,2,4];
    % split trials into groups (num groups = opt.numFolds)
    [points,~,IC] = unique(points); 
    data_grp = nan(size(points));
    shift_by = 0; % to make sure equal numbers of trials end up in each fold
    % make sure all folds have roughly equal numbers of points from every rewsize
    for i = 1:numel(xval_opt.rew_size)
        keep_this = all_concat_rewsize_noPrewRew == xval_opt.rew_size(i);
        data_grp_this = repmat(circshift(1:xval_opt.numFolds,shift_by),1,ceil(sum(keep_this)/xval_opt.numFolds)*xval_opt.numFolds);
        data_grp(keep_this) = data_grp_this(1:sum(keep_this)); % assign folds 1:10
        shift_by = shift_by - mod(sum(keep_this),xval_opt.numFolds); % shift which fold is getting fewer trials
    end
    
    foldid = data_grp(IC)';  
    threshold_step = .05;
    thresholds = 0:threshold_step:1; 
    pc_ranges = 1:10;
    
    new_xval = true;
    if new_xval == true 
%         set up datastructures to measure classification fidelity
        accuracies = nan(numel(pc_ranges),xval_opt.numFolds,numel(thresholds));
        precisions = nan(numel(pc_ranges),xval_opt.numFolds,numel(thresholds));
        TP_rates = nan(numel(pc_ranges),xval_opt.numFolds,numel(thresholds));
        FP_rates = nan(numel(pc_ranges),xval_opt.numFolds,numel(thresholds)); 
        ROC_AUC = nan(numel(pc_ranges),xval_opt.numFolds); 
        PR_AUC = nan(numel(pc_ranges),xval_opt.numFolds);
        
        for pcIdx = 1:numel(pc_ranges) 
            last_pc = pc_ranges(pcIdx);
            
            % Iterate over folds to use as test data
            for fIdx = 1:xval_opt.numFolds
                % separate training and test data
                data_train = all_concat_PCs_noPreRew(1:last_pc,foldid~=fIdx);
                labels_train = all_concat_labels_noPreRew(foldid~=fIdx);
                labels_test = all_concat_labels_noPreRew(foldid==fIdx);
                data_test = all_concat_PCs_noPreRew(1:last_pc,foldid==fIdx);
                
                % now fit logistic regression to our training data
                [B,~,~] = mnrfit(data_train',labels_train);
                pi_test = mnrval(B,data_test');
                
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
                    accuracies(pcIdx,fIdx,tIdx) = accuracy;
                    precisions(pcIdx,fIdx,tIdx) = precision;
                    TP_rates(pcIdx,fIdx,tIdx) = TP_rate;
                    FP_rates(pcIdx,fIdx,tIdx) = FP_rate;
                end 
                
                ROC_AUC_dx = -squeeze(diff(FP_rates(pcIdx,fIdx,:)));
                ROC_AUC(pcIdx,fIdx) = sum(ROC_AUC_dx .* squeeze(TP_rates(pcIdx,fIdx,1:end-1))); 
                PR_AUC_dx = -squeeze(diff(TP_rates(pcIdx,fIdx,:)));
                PR_AUC(pcIdx,fIdx) = sum(PR_AUC_dx(~isnan(precisions(pcIdx,fIdx,1:end-1))) .* squeeze(precisions(pcIdx,fIdx,~isnan(precisions(pcIdx,fIdx,1:end-1)))));
            end 
            if mod(pcIdx,2) == 0
                fprintf("PC 1:%i Complete \n",pcIdx) 
            end
        end 
        
        % Now repeat for velocity to have comparison
        accuracies_vel = nan(xval_opt.numFolds,numel(thresholds));
        precisions_vel = nan(xval_opt.numFolds,numel(thresholds));
        TP_rates_vel = nan(xval_opt.numFolds,numel(thresholds));
        FP_rates_vel = nan(xval_opt.numFolds,numel(thresholds));
        ROC_AUC_vel = nan(xval_opt.numFolds,1); 
        PR_AUC_vel = nan(xval_opt.numFolds,1);
        for fIdx = 1:xval_opt.numFolds
            % separate training and test data
            data_train = all_concat_vel_noPreRew(foldid~=fIdx);
            labels_train = all_concat_labels_noPreRew(foldid~=fIdx);
            data_test = all_concat_vel_noPreRew(foldid==fIdx);  
            labels_test = all_concat_labels_noPreRew(foldid==fIdx);
            
            % now fit logistic regression to our training data
            [B,~,~] = mnrfit(data_train',labels_train);
            pi_test = mnrval(B,data_test');
            
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
                accuracies_vel(fIdx,tIdx) = accuracy;
                precisions_vel(fIdx,tIdx) = precision;
                TP_rates_vel(fIdx,tIdx) = TP_rate;
                FP_rates_vel(fIdx,tIdx) = FP_rate;
            end 
            
            ROC_AUC_dx_vel = -squeeze(diff(FP_rates_vel(fIdx,:)));
            ROC_AUC_vel(fIdx) = sum(ROC_AUC_dx_vel .* TP_rates_vel(fIdx,1:end-1)); 
            PR_AUC_dx_vel = -squeeze(diff(TP_rates_vel(fIdx,:)));
            PR_AUC_vel(fIdx) = sum(PR_AUC_dx_vel(~isnan(precisions_vel(fIdx,1:end-1))) .* precisions_vel(fIdx,~isnan(precisions_vel(fIdx,1:end-1))));
        end 
        
        % Repeat for mean ramp up neurons
        accuracies_ramp = nan(xval_opt.numFolds,numel(thresholds));
        precisions_ramp = nan(xval_opt.numFolds,numel(thresholds));
        TP_rates_ramp = nan(xval_opt.numFolds,numel(thresholds));
        FP_rates_ramp = nan(xval_opt.numFolds,numel(thresholds));
        ROC_AUC_ramp = nan(xval_opt.numFolds,1); 
        PR_AUC_ramp = nan(xval_opt.numFolds,1);
        for fIdx = 1:xval_opt.numFolds
            % separate training and test data
            data_train = all_concat_meanRamp_upAll_noPreRew(foldid~=fIdx);
            labels_train = all_concat_labels_noPreRew(foldid~=fIdx);
            data_test = all_concat_meanRamp_upAll_noPreRew(foldid==fIdx);  
            labels_test = all_concat_labels_noPreRew(foldid==fIdx);
            
            % now fit logistic regression to our training data
            [B,~,~] = mnrfit(data_train',labels_train);
            pi_test = mnrval(B,data_test');
            
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
                accuracies_ramp(fIdx,tIdx) = accuracy;
                precisions_ramp(fIdx,tIdx) = precision;
                TP_rates_ramp(fIdx,tIdx) = TP_rate;
                FP_rates_ramp(fIdx,tIdx) = FP_rate;
            end 
            
            ROC_AUC_dx_ramp = -squeeze(diff(FP_rates_ramp(fIdx,:)));
            ROC_AUC_ramp(fIdx) = sum(ROC_AUC_dx_ramp .* TP_rates_ramp(fIdx,1:end-1)); 
            PR_AUC_dx_ramp = -squeeze(diff(TP_rates_ramp(fIdx,:)));
            PR_AUC_ramp(fIdx) = sum(PR_AUC_dx_ramp(~isnan(precisions_ramp(fIdx,1:end-1))) .* precisions_ramp(fIdx,~isnan(precisions_ramp(fIdx,1:end-1))));
        end 
    end
    
    % visualize results with AUROC and Precision-Recall Curve
    for pcIdx = [1,5,10]
%         figure(figcounter)
%         last_pc = pc_ranges(pcIdx);
%         errorbar(thresholds,squeeze(mean(accuracies(pcIdx,:,:))),1.96 * squeeze(std(accuracies(pcIdx,:,:))),'linewidth',1.5) 
%         hold on
%         xlabel("Threshold")
%         ylabel("Mean Test Set Accuracy")
%         title("10-fold Test Accuracy Across Thresholds")
        figure(figcounter)
        subplot(1,2,1)
        errorbar(squeeze(mean(FP_rates(pcIdx,:,:))),squeeze(mean(TP_rates(pcIdx,:,:))),1.96 * squeeze(std(TP_rates(pcIdx,:,:))),'linewidth',1.5)
        hold on
        xlabel("Mean False Positive Rate Across Folds")
        ylabel("Mean True Positive Rate Across Folds")
        title(sprintf("%s Receiver Operator Characteristic Curve",session_title))
        subplot(1,2,2)
        errorbar(squeeze(mean(TP_rates(pcIdx,:,:))),squeeze(mean(precisions(pcIdx,:,:))),1.96 * squeeze(std(precisions(pcIdx,:,:))),'linewidth',1.5)
        hold on
        xlabel("Mean True Positive Rate Across Folds")
        ylabel("Mean Precision Across Folds")
        title(sprintf("%s Precision Recall Curve",session_title))
    end 
    
    figure(figcounter)  
    subplot(1,2,1)
    errorbar(mean(FP_rates_vel),mean(TP_rates_vel),1.96 * std(TP_rates_vel),'linewidth',1.5)  
    errorbar(mean(FP_rates_ramp),mean(TP_rates_ramp),1.96 * std(TP_rates_ramp),'linewidth',1.5)  
    plot([0,1],[0,1],'k--','linewidth',1.5) 
    ylim([0,1])
    legend("PC 1:1","PC 1:5","PC 1:10","Velocity","Naive Performance") 
    subplot(1,2,2) 
    errorbar(mean(TP_rates_vel),mean(precisions_vel),1.96 * std(precisions_vel),'linewidth',1.5) 
    errorbar(mean(TP_rates_ramp),mean(precisions_ramp),1.96 * std(precisions_ramp),'linewidth',1.5)  
    yline(.5,'k--','linewidth',1.5)
    legend("PC 1:1","PC 1:5","PC 1:10","Velocity","Mean Ramping Activity","Naive Performance") 
    ylim([0,1])
    
    % Now plot AUC 
    figure(figcounter + 1) 
    subplot(1,2,1)
    errorbar(pc_ranges,mean(ROC_AUC,2),1.96 * std(ROC_AUC,[],2),'linewidth',1.5)
    hold on 
    yline(mean(ROC_AUC_vel),'k--','linewidth',1.5)   
    yline(mean(ROC_AUC_vel) + 1.95 * std(ROC_AUC_vel),'k:','linewidth',1.5) 
    yline(mean(ROC_AUC_vel) - 1.95 * std(ROC_AUC_vel),'k:','linewidth',1.5) 
    title(sprintf("%s AUROC Forward Search",session_title)) 
    legend("AUROC for PCs","AUROC for Velocity") 
    xlabel("PCs Used In Logistic Regression") 
    ylabel("AUROC") 
    ylim([0,1])
    subplot(1,2,2)
    errorbar(pc_ranges,mean(PR_AUC,2),1.96 * std(PR_AUC,[],2),'linewidth',1.5) 
    hold on 
    yline(mean(PR_AUC_vel),'k--','linewidth',1.5) 
    yline(mean(PR_AUC_vel) + 1.95 * std(PR_AUC_vel),'k:','linewidth',1.5) 
    yline(mean(PR_AUC_vel) - 1.95 * std(PR_AUC_vel),'k:','linewidth',1.5) 
    title(sprintf("%s AUPR Forward Search",session_title)) 
    legend("AUPR for PCs","AUPR for Velocity") 
    xlabel("PCs Used In Logistic Regression")
    ylabel("AUPR") 
    ylim([0,1]) 
    
    fprintf("Session %s Complete \n",session_title)
    
    figcounter = figcounter + 2;
end 