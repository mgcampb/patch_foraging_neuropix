function [msLickApex, msGroom] = detect_small_lick_by_deflection_malcolm(vLick, vThres, samp_rate)
% find deflection points that is above threshold (vLickThres), and the peak
% is greater than another criterion (vApexThres).
% capacitive lick detector gives the signal that always goes back to the
% baseline, but IR-based lick detector does not. I need to detect
% deflection point. 
% HRK 6/30/2015 

% edited (streamlined)
% added enforced lick refractory period of 80 ms
% MGC 4/1/2020

if is_arg('vThres')
    vLickThres = vThres;
else
    vLickThres = 0.147;
end
if ~is_arg('samp_rate')
    samp_rate = 1000;
end
vGroomThres = 4.3; % 4.0 -> 4.3 6/23/2016 HRK

% smooth using moving average of windows size 6.
% This is not good for onset detection, but needed for deflection detection
assert(any(size(vLick) == 1) && length(size(vLick)) == 2)
vLick = vLick(:);
sm_vLick = conv2(vLick, ones(30,1)/30, 'same');

% detect licks
apex = detect_deflection(sm_vLick, vLickThres, 'convex'); % vLickThres=0 gives the same results
if isempty(apex)
    disp('WARNING: No licks detected');
    msLickApex = []; msGroom = [];
    return
end

% eliminate small apex
msGroom = apex( sm_vLick(apex) >= vGroomThres );
msLickApex = apex( sm_vLick(apex) < vGroomThres );

% enforce a refractory period of 80ms for licks
keep_lick = false(size(msLickApex));
keep_lick(1) = true;
ref_lick = msLickApex(1);
for i = 2:numel(msLickApex)
    lick_diff = msLickApex(i)-ref_lick;
    if lick_diff > 0.08*samp_rate % 80 ms enforced refractory period
        keep_lick(i) = true;
        ref_lick = msLickApex(i);
    end
end
msLickApex = msLickApex(keep_lick);

if length(msLickApex) / (max( apex )/samp_rate) > 8 % average lick rate cannot exceed 8
    disp('ERROR: Lick rate exceeds 8. Returning nans;');
    msLickApex = nan; msGroom = nan;
    return;
end

end