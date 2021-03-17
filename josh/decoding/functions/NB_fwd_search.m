function [mi_cumulative,mae_cumulative,rmse_cumulative,timecourse_results] = NB_fwd_search(population,search_depth,timecourse_save_steps,mIdx,i_session,iVar,iRewsize,iFeature,X_dataset,y_dataset,models,xval_table,dataset_opt)
% Perform forward search, adding cells one by one to trained naive bayes
% decoder
    
    cells_left = population; % the PCs that are left to pick
    cells_picked = []; % store PCs that we are keeping in forward search 

    % 1) Grab the relevant X, y, and xval information
    this_X = X_dataset{mIdx}{i_session}{iVar}{iRewsize}{iFeature};  
    nNeurons = size(this_X{1},1); 
    these_models = models{mIdx}{i_session}{iVar}{iRewsize}{iFeature};
    foldID = xval_table{mIdx}{i_session}.FoldID(xval_table{mIdx}{i_session}.Rewsize == dataset_opt.rewsizes(iRewsize));
    % Get y_true in proper order by folds
    this_y_true = y_dataset{mIdx}{i_session}{iVar}{iRewsize};  
    this_y_true_tmp = cell(dataset_opt.numFolds,1);
    for kFold = 1:dataset_opt.numFolds 
        kFold_y_true = this_y_true(foldID == kFold);  
        this_y_true_tmp{kFold} = cat(2,kFold_y_true{:}); 
    end 
    this_y_true = cat(2,this_y_true_tmp{:}); 

    % 2) Loop over cells to include  
    mi_cumulative = nan(search_depth,1); 
    mae_cumulative = nan(search_depth,1); 
    rmse_cumulative = nan(search_depth,1); 
    timecourse_results = cell(length(timecourse_save_steps),1); % $
    timecourse_counter = 1; % $

    h1 = waitbar(0,sprintf('Performing forward search with depth %i',search_depth));

    for added_cell = 1:search_depth
        mi_cells_left = nan(length(cells_left),1);  
        mae_cells_left = nan(length(cells_left),1);  
        rmse_cells_left = nan(length(cells_left),1);
        h2=waitbar(0,sprintf('Iterating to find cell %i',added_cell));
        % Change position of second bar so the is not overlap
        pos_w1=get(h1,'position');
        pos_w2=[pos_w1(1) pos_w1(2)-pos_w1(4) pos_w1(3) pos_w1(4)];
        set(h2,'position',pos_w2,'doublebuffer','on')

        for i_cell = 1:numel(cells_left) 
            cells_picked_tmp = [cells_picked cells_left(i_cell)];
            nan_cells = setdiff(1:nNeurons,cells_picked_tmp); % cells to set to nan

            % 3) Cross-validated prediction, setting rows from not included cells = nan
            i_y_hat_tmp = cell(numel(dataset_opt.numFolds),1);
            for kFold = 1:dataset_opt.numFolds
                X_test = this_X(foldID == kFold);
                X_test = cat(2,X_test{:})';  
                X_test(:,nan_cells) = NaN; 

                i_y_hat = predict(these_models{kFold},X_test);

                i_y_hat_tmp{kFold} = i_y_hat;
            end
            i_y_hat_full = cat(1,i_y_hat_tmp{:})';  

            mi_cells_left(i_cell) = MI_confusionmat(confusionmat(i_y_hat_full,this_y_true)); 
            mae_cells_left(i_cell) = nanmean(abs(i_y_hat_full-this_y_true)); 
            rmse_cells_left(i_cell) = sqrt(nanmean((i_y_hat_full-this_y_true).^2)); 
            waitbar(i_cell / numel(cells_left),h2)
        end
        close(h2);

        % pick cell that predicts w/ highest MI
        [max_mi,cell_new_ix] = max(mi_cells_left);  
        mi_cumulative(added_cell) = max_mi; % save mutual information that got us
        mae_cumulative(added_cell) =  mae_cells_left(cell_new_ix);
        rmse_cumulative(added_cell) = rmse_cells_left(cell_new_ix); 
        cell_new = cells_left(cell_new_ix); % grab the cell
        cells_left = setdiff(cells_left,cell_new); % remove it from the cells left to add
        cells_picked = [cells_picked cell_new]; % add it to cells picked
        waitbar(added_cell / search_depth,h1) 
        
        % If we are on a timecourse save step, take metrics on timecourse of decoding 
        if ismember(added_cell,timecourse_save_steps) % $ 
            timecourse_results{timecourse_counter} = save_timecourse_metrics(cells_picked,nNeurons,timecourse_counter,dataset_opt,this_X,this_y_true,foldID,these_models);
            timecourse_counter = timecourse_counter + 1; 
        end 
    end
    close(h1); 

end

