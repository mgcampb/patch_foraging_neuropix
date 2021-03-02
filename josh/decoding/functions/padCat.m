function X_full = padCat(X_cell,sessionIx)
% Concatenate neural data (from some folds) across sessions with NaN padding 
% Requires trials to be sorted by session
    t_lens = cellfun(@(x) size(x,2),X_cell); % sum trial lengths across sessions
%     [s_nNeurons,~,session_labels] = unique(cellfun(@(x) size(x,1),X_cell)); 
    % get the session lengths
    session_lengths = arrayfun(@(x) sum(cellfun(@(y) size(y,2),X_cell(sessionIx == x))),unique(sessionIx)); 
    session_ix_starts = [0 cumsum(session_lengths)']; % w/ the lengths, get starting index so we can drop in data at proper rows
    session_trial_starts = [cell2mat(arrayfun(@(x) find(sessionIx==x,1),unique(sessionIx),'un',false))',length(sessionIx)+1];   
    s_nNeurons = arrayfun(@(x) size(X_cell{x},1),session_trial_starts(1:end-1)); 
    session_neuron_starts = [0 cumsum(s_nNeurons)]; 
    X_full = nan(sum(s_nNeurons),sum(t_lens)); 
    % iterate over sessions, and fill in values where needed
    for s_ix = 1:(numel(session_ix_starts)-1) 
        X_session_cell = X_cell(session_trial_starts(s_ix):session_trial_starts(s_ix+1)-1); 
        X_session = cat(2,X_session_cell{:});
        X_full(session_neuron_starts(s_ix)+1:session_neuron_starts(s_ix+1),session_ix_starts(s_ix)+1:session_ix_starts(s_ix+1)) = X_session;
    end
end