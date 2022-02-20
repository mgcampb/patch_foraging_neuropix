%% See what's going on with rand index
% ref: https://faculty.washington.edu/kayee/pca/supp.pdf

p1 = [1 2 3 3 3 3];
p2 = [1 2 2 2 2 2];

N = length(p1);
[~, ~, p1] = unique(p1);
N1 = max(p1);
[~, ~, p2] = unique(p2);
N2 = max(p2);

clear n
% Create the contingency matrix n
%   n(i,:) = # objects in class U(i)
%   n(:,j) = # objects in class V(i)
for i=1:1:N1
    for j=1:1:N2
        G1 = find(p1==i);
        G2 = find(p2==j);
        n(i,j) = length(intersect(G1,G2));
    end
end

% RI = (a + d) / (a + b + c + d)
%  a = # pairs of objects in same class in U and V = global nchoose2 sum
%  d = # pairs of objects in diff class in U and V = (total objects choose 2) - a - b - c
%  b = # pairs of objects in same class in U but diff class in V = sum(row sums choose 2) - a
%  c = # pairs of objects in diff class in U but same class in V = sum(col sums choose 2) - a

a = 0;
for i = 1:N1
    for j = 1:N2
        a = a + nchoosek2(n(i,j),2);
    end
end

row_sum = sum(n,2);
row_choosesum = 0;
for i = 1:N1
    row_choosesum = row_choosesum + nchoosek2(row_sum(i),2);
end

col_sum = sum(n,1);
col_choosesum = 0;
for j = 1:N2
    col_choosesum = col_choosesum + nchoosek2(col_sum(j),2);
end

b = row_choosesum - a;
c = col_choosesum - a;
d = nchoosek2(sum(n(:)),2) - a - b - c;

ri = (a + d) / (a + b + c + d);
fprintf("manually-calculated rand index: %.3f\n",ri)
fprintf("ss-calculated rand index: %.3f\n\n",rand_index(p1,p2))

E_ri = (row_choosesum * col_choosesum) /  nchoosek2(sum(n(:)),2) ;
other_denom_term = .5 * (row_choosesum + col_choosesum);
ari = (a - E_ri) / (other_denom_term - E_ri);
fprintf("manually-calculated adjusted rand index: %.3f \n",ari)
fprintf("ss-calculated adjusted rand index: %.3f\n\n",rand_index(p1,p2,'adjusted'))

%% Test congruency and speed between basic and sum of squares method
n_tests = 100;
max_label = 10;
n_labels = [10 50 100 1000 5000];

basic_time = nan(numel(n_labels),1);
fancy_time = nan(numel(n_labels),1);
for i_n_labels = 1:numel(n_labels)
    this_n_labels = n_labels(i_n_labels);
    
    p1 = randi(max_label,[n_tests this_n_labels]);
    p2 = randi(max_label,[n_tests this_n_labels]);
    
    basic_ari = nan(n_tests,1);
    tic
    for i_test = 1:n_tests
        basic_ari(i_test) = rand_index(p1(i_test,:),p2(i_test,:),'adjusted');
    end
    basic_time(i_n_labels) = toc;
    
    fancy_ari = nan(n_tests,1);
    tic
    for i_test = 1:n_tests
        fancy_ari(i_test) = rand_index(p1(i_test,:),p2(i_test,:),'adjusted','fancy');
    end
    fancy_time(i_n_labels) = toc;
    
    fprintf("All the same: %d \n",all(basic_ari == fancy_ari))
end
% result: they're actually basically the same

%% Now compare ARI + bootstrapping vs BIC/AIC vs nearest neighbor hit rate on some synthetic gaussian data

% generate synthetic data
n_clusters_true = 3;
n_samples = 500;
true_clusters = sort(randi(n_clusters_true , [n_samples 1]));

rng(1)
mean_scale = 4;
diag_scale = 1;
offdiag_scale = .1;
mu = mean_scale * [rand(n_clusters_true,1) rand(n_clusters_true,1)];
diag_rand = rand();
sigma = arrayfun(@(i) min(diag_rand * diag_scale,offdiag_scale * rand()) * (eye(2) == 0) + diag_scale * diag_rand * eye(2),1:n_clusters_true,'un',0);
data = nan(n_samples,2);
for i_cluster = 1:n_clusters_true
    data(true_clusters == i_cluster,:) = mvnrnd(mu(i_cluster,:),sigma{i_cluster},length(find(true_clusters == i_cluster)));
end

gscatter(data(:,1),data(:,2),true_clusters)
title("Synthetic Gaussian 3-Cluster Data")
set(gca,'fontsize',14)
%% Fit Gaussian mixture models

gmm_opt.replicates = 20;
options = statset('MaxIter',1000);
n_repeats = 50;
n_clusters = [3 4];
lambdas = [0];

BIC = nan(numel(lambdas),numel(n_clusters));
AIC = nan(numel(lambdas),numel(n_clusters));
b = waitbar(0,'Performing BMS Analysis');
compute_counter = 0;
for i_lambda = 1:numel(lambdas)
    this_lambda = lambdas(i_lambda);
    for i_n_clusters = 1:numel(n_clusters)
        this_n_clusters = n_clusters(i_n_clusters);
        % fit model
        GMM = fitgmdist(data,this_n_clusters,'RegularizationValue',this_lambda,'replicates',gmm_opt.replicates,'options',options);
        BIC(i_lambda,i_n_clusters) = GMM.BIC;
        AIC(i_lambda,i_n_clusters) = GMM.AIC;
        gmm_idx = cluster(GMM,data);
        subplot(numel(lambdas),numel(n_clusters),numel(n_clusters) * (i_lambda - 1) + i_n_clusters)
        gscatter(data(:,1),data(:,2),gmm_idx,[],[],[],0)
        compute_counter = compute_counter + 1;
        waitbar(compute_counter / (numel(lambdas) * numel(n_clusters)));
        if i_n_clusters == 1
            ylabel(sprintf("λ = %.2f",this_lambda),'fontsize',14)
        end
        if i_lambda == numel(lambdas)
            xlabel(sprintf("%i Clusters",this_n_clusters),'fontsize',14)
        end
    end
end
close(b);

% Now perform rand index bootstrapping procedure
ari = gmm_ari_gridSearch(data,n_repeats,lambdas,n_clusters,gmm_opt,options);


%% Visualize BIC and AIC over clusters and reg values

figure()
subplot(1,3,1)
imagesc((BIC - min(BIC,[],2)) ./ (max(BIC,[],2) - min(BIC,[],2)))
xticks(1:numel(n_clusters));xticklabels(n_clusters)
yticks(1:numel(lambdas));yticklabels(lambdas)
title("BIC over choices of λ and # clusters")
ylabel("λ")
xlabel("# clusters")
colorbar()

set(gca,'fontsize',14)

subplot(1,3,2)
imagesc((AIC - min(AIC,[],2)) ./ (max(AIC,[],2) - min(AIC,[],2)))
xticks(1:numel(n_clusters));xticklabels(n_clusters)
yticks(1:numel(lambdas));yticklabels(lambdas)
title("AIC over choices of λ and # clusters")
xlabel("# clusters")
colorbar()
set(gca,'fontsize',14)

subplot(1,3,3)
imagesc(nanmean(ari,3))
xticks(1:numel(n_clusters));xticklabels(n_clusters)
xlabel("# Clusters")
yticks(1:numel(lambdas));yticklabels(lambdas)
ylabel("λ")
title("Mean ARI over choices of λ and # Clusters")
colorbar()

set(gca,'fontsize',14)

%% Visualize ARI
close all
figure(); hold on

for i_lambda = 2 % :numel(lambdas)
    %     subplot(1,numel(lambdas),i_lambda);hold on
    for i_n_clusters = 1:numel(n_clusters)
        [f,xi] = ksdensity(squeeze(ari(i_lambda,i_n_clusters,:)));
        plot(xi,f,'linewidth',1.5);
    end
    title(sprintf("λ = %.2f",lambdas(i_lambda)))
end
% subplot(1,numel(n_clusters),1)
legend(arrayfun(@(i) sprintf("%i Clusters",n_clusters(i)),(1:numel(n_clusters))))
ylabel("Distribution of ARI bootstrap sample pairs",'fontsize',14)
xlabel("ARI")
set(gca,'fontsize',14)

% figure()
% imagesc(nanmean(ari,3))
% xticks(1:numel(n_clusters));xticklabels(n_clusters)
% xlabel("# Clusters")
% yticks(1:numel(lambdas));yticklabels(lambdas)
% ylabel("λ")
% title("Mean ARI over choices of λ and # Clusters")
% set(gca,'fontsize',14)

%% Calculate significance of difference change in ARI

seq_p = nan(numel(lambdas),(numel(n_clusters)-1));
for i_lambda = 1:numel(lambdas)
    for i_n_clusters = 1:(numel(n_clusters)-1)
        [h,this_p] = ttest(squeeze(ari(i_lambda,i_n_clusters,:)),squeeze(ari(i_lambda,i_n_clusters + 1,:)));
        seq_p(i_lambda,i_n_clusters) = this_p;
    end
end
imagesc(seq_p)

%% functions
function c = nchoosek2(a,b)
if a > 1
    c = nchoosek(a,b);
else
    c = 0;
end
end