function y = my_zscore(x)
% zscores over rows
% MGC 4/14/2020
y = (x-mean(x,2))./repmat(std(x,[],2),1,size(x,2));
end

