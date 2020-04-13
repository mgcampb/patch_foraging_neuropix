function [idx_apex v_apex] = detect_deflection(train, threshold, mode)
% HRK 6/30/2015 

% dx/dt
d_train = diff(train);

% if mode is not set, use sign of threshold to set it.
if ~exist('mode','var') || isempty(mode)
    if threshold > 0
        mode = 'convex';
    elseif threshold < 0
        mode = 'concave';
    end
end

switch(mode)
    case 'convex'
        % because of digitization, I saw several missing defection in a
        % datafile if I do not use '='. for small lick detection algorithm.
        bApex = d_train(1:end-1) > 0 & d_train(2:end) <= 0;

    case 'concave'
        bApex = d_train(1:end-1) < 0 & d_train(2:end) >= 0;

    case 'both'
        bApex = d_train(1:end-1) > 0 & d_train(2:end) <= 0 | ...
            d_train(1:end-1) < 0 & d_train(2:end) >= 0;
        
    otherwise
        error('Unknown mode, %s', mode);
end

% match time
bApex = [0; bApex; 0];

% apply threahold. do nothiing if threshold is NaN
if threshold > 0
    bApex = bApex & (train > threshold);
elseif threshold < 0
    bApex = bApex & (train < threshold);
end

idx_apex = find(bApex);
v_apex = train(idx_apex);