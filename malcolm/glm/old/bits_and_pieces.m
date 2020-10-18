% hfig = figure; hold on;
% hfig.Name = 'discrete event basis functions';
% for i = 1:opt.nbasis
%     plot(t_basis,bas(i,:));
% end
% xlabel('time (sec)');
% title('discrete event basis functions');
% save_figs(paths.figs,hfig,'png');


% % PATCH CUE 1: patch cue to (patch stop or patch leave)
% patch_cue1 = zeros(numel(t),1);
% trigs = [dat.patchCL; dat.patchCSL(:,1:2)];
% for i = 1:size(trigs,1)
%     patch_cue1(t-trigs(i,1)>=0 & t<trigs(i,2)) = 1;
% end
% X_this = patch_cue1;
% A = [A, {X_this}];
% grp_name = [grp_name,'PatchCue1'];
% var_name = [var_name,'PatchCue1'];
% 
% % PATCH CUE 2: patch stop to patch leave
% patch_cue2 = zeros(numel(t),1);
% trigs = dat.patchCSL(:,2:3);
% for i = 1:size(trigs,1)
%     patch_cue2(t-trigs(i,1)>=0 & t<trigs(i,2)) = 1;
% end
% X_this = patch_cue2;
% A = [A,{X_this}];
% grp_name = [grp_name,'PatchCue2'];
% var_name = [var_name,'PatchCue2'];




% DV from model 3
% DV3 = t_on_patch-1.7172*tot_rew; % this the fit for m80 0317



% dropout each predictor individually
X_dropout = {};
for i = 1:numel(A)
    X_dropout{i} = [];
    for j = 1:numel(A)
        if j~=i
            X_dropout{i} = [X_dropout{i} A{j}];
        end
    end
end

% get standard errors on coefficients
X_this = [ones(size(X_final,1),1) X_final];    
mu = X_this*beta_this;    
y_this = score_final(:,pIdx);
sigma2 = mean((y_this-mu).^2);
W = diag(mu.^2/sigma2);   
V = inv(X_this'*W*X_this);
se_all(:,pIdx) = sqrt(diag(V));
R2(pIdx) = 1-sigma2/mean((y_this-mean(y_this)).^2);