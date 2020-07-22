function [log_llh] = compute_llh(A,n,param)

r = exp(A * param); n = reshape(n,numel(n),1); meanFR = nanmean(n);

log_llh_model = nansum(r-n.*log(r)+log(factorial(n)))/sum(n);
log_llh_mean = nansum(meanFR-n.*log(meanFR)+log(factorial(n)))/sum(n);
log_llh = (-log_llh_model + log_llh_mean);
log_llh = log(2)*log_llh;

return
