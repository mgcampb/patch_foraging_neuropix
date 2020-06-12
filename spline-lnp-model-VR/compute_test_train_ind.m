function [train_ind,test_ind] = compute_test_train_ind(numFolds,numPts,T)

numSeg = ceil(T/(numFolds*numPts));
oneSeg = ones(numPts,1)*(1:numFolds);
new_ind = repmat(oneSeg(:),numSeg,1);
new_ind = new_ind(1:T);

test_ind = cell(numFolds,1);
train_ind = cell(numFolds,1);
for k = 1:numFolds
    test_ind{k} = find(new_ind == k);
    train_ind{k} = setdiff(1:T,test_ind{k});
end

return