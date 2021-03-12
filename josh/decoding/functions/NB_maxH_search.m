function [mi_cumulative,mae_cumulative,timecourse_results,peak_distns] = NB_maxH_search(population,n_searches,search_depth,timecourse_save_steps,...
                                                                            mIdx,i_session,iVar,iRewsize,iFeature,...
                                                                            X_dataset,y_dataset,models,xval_table,dataset_opt,peak_times)
% NB_MAXH_SEARCH Add cells one-by-one to Naive Bayes decoding, selecting new
%               cells to maximize entropy of peak time distribution

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
    mi_cumulative = nan(n_searches,search_depth);
    mae_cumulative = nan(n_searches,search_depth); 
    peak_distns = cell(n_searches,1); 
    timecourse_results = cell(n_searches,1);  

    h0 = waitbar(0,sprintf('Performing %i iterations of max H cell addition',n_searches));
    for i_search = 1:n_searches 
        h1 = waitbar(0,sprintf('Performing max H cell additions with depth %i',search_depth));
        % Change position of second bar so the is not overlap
        pos_w1=get(h0,'position');
        pos_w2=[pos_w1(1) pos_w1(2)-pos_w1(4) pos_w1(3) pos_w1(4)];
        set(h1,'position',pos_w2,'doublebuffer','on')
        
        % Save timecourse if on the first search 
        timecourse_results{i_search} = cell(length(timecourse_save_steps),1); % $ 
        peak_distns{i_search} = cell(search_depth,1); 
        timecourse_counter = 1;
        cells_left = population; % cells left to pick
        cells_picked = []; % store cells that we are keeping in forward search 
        
        for added_cell = 1:search_depth % add search_depth cells  
            H_cells_left = nan(length(cells_left),1); % calculate resulting H per added cells_picked 
            for i_cell = 1:numel(cells_left)
                cells_picked_tmp = [cells_picked cells_left(i_cell)];   
                H_cells_left(i_cell) = calc_shannonH(peak_times(cells_picked_tmp));
            end
            
            % Add cell to maximize entropy of peak time distn
            max_H = max(H_cells_left); % max entropy value 
            max_H_cells = find(H_cells_left == max_H); 
            cell_new_ix = max_H_cells(randi(length(max_H_cells))); % of the cells that produce the same max_H, choose a random one
            cell_new = cells_left(cell_new_ix); 
            cells_left = setdiff(cells_left,cell_new); % remove it from the cells left to add
            cells_picked = [cells_picked cell_new]; % add it to cells picked  
            peak_distns{i_search}{added_cell} = peak_times(cells_picked); 

            nan_cells = setdiff(1:nNeurons,cells_picked); % cells to set to nan

            % Perform Cross-validated prediction, setting rows from not included cells = nan
            i_y_hat_tmp = cell(numel(dataset_opt.numFolds),1);
            for kFold = 1:dataset_opt.numFolds
                X_test = this_X(foldID == kFold);
                X_test = cat(2,X_test{:})';
                X_test(:,nan_cells) = NaN;

                i_y_hat = predict(these_models{kFold},X_test);

                i_y_hat_tmp{kFold} = i_y_hat;
            end
            i_y_hat_full = cat(1,i_y_hat_tmp{:})';

            mi_cumulative(i_search,added_cell) = MI_confusionmat(confusionmat(i_y_hat_full,this_y_true));
            mae_cumulative(i_search,added_cell) = nanmean(abs(i_y_hat_full-this_y_true));
            
            % If we are on a timecourse save step, take metrics on timecourse of decoding
            if ismember(added_cell,timecourse_save_steps) % $
                timecourse_results{i_search}{timecourse_counter} = save_timecourse_metrics(cells_picked,nNeurons,timecourse_counter,dataset_opt,this_X,this_y_true,foldID,these_models);
                timecourse_counter = timecourse_counter + 1;
            end 
            
            waitbar(added_cell / search_depth,h1)
        end
        close(h1); 
        waitbar(i_search / n_searches,h0)
    end
    close(h0)

end

