function [X,cpts_all] = spline_spike_coupling(spiketrain,cpts,s)


%% This is adapted from Uri Eden's spline lab.

% spiketrain = the # of spikes over time
% cpts = control points
% s = tension parameters

% S matrix
S = [-s 2-s s-2 s;2*s s-3 3-2*s -s;-s 0 s 0;0 1 0 0];

% Q - max time in the past and future
Q_past = cpts(1)+1;
Q_future = cpts(end);

% Construct spline regressors
bin1 = cpts(2) - cpts(1); bin2 = cpts(end)-cpts(end-1);
cpts_all = [cpts(1)-bin1 cpts cpts(end)+bin2];

num_c_pts = length(cpts_all);  %number of control points in total

p_mat = nan(numel(Q_past:Q_future),num_c_pts);
counter = 0;
for tau = Q_past:Q_future
    counter = counter + 1;
    nearest_c_pt_index = max(find(cpts_all<tau));
    nearest_c_pt_time = cpts_all(nearest_c_pt_index);
    next_c_pt_time = cpts_all(nearest_c_pt_index+1);
    u = (tau-nearest_c_pt_time)/(next_c_pt_time-nearest_c_pt_time);
    p =[u^3 u^2 u 1]*S;
    p_mat(counter,:) = [zeros(1,nearest_c_pt_index-2) p zeros(1,num_c_pts-4-(nearest_c_pt_index-2))];
end

X = zeros(length(spiketrain),length(cpts_all));
%for each timepoint, calculate the corresponding row of the glm input matrix
for i=1:length(spiketrain)
    
    % compute the past spiking history
    if i >= abs(Q_past)+1
        past = spiketrain(i+Q_past:i-1); % order: 1 bin, 2 bins, 3 ...
    else
        past = [zeros(abs(Q_past)-i+1,1)' spiketrain(1:i-1)];
    end
    
    % compute the future spiking history
    if i <= length(spiketrain) - Q_future
        future = spiketrain(i:i+Q_future);
    else
        future = [spiketrain(i:end) zeros(1,Q_future + i - length(spiketrain)) ];
    end
    
    % fill in the X matrix, with the right # of zeros on either side
    X(i,:) = [past future]*p_mat;
    
end

