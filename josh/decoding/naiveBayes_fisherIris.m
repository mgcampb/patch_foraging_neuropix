%% Get familiar with Naive bayes decoding on fisher iris dataset 
% load data
load fisheriris
X = meas(:,3:4); % take data as last two measures
Y = species;
% tabulate(Y) 

%% fit naive bayes model
Mdl = fitcnb(X,Y,'ClassNames',{'setosa','versicolor','virginica'});  

%% check out model distribution estimates
setosaIndex = strcmp(Mdl.ClassNames,'setosa');
estimates = Mdl.DistributionParameters{setosaIndex,1} 

%% Visualize fitted distributions  
figure
gscatter(X(:,1),X(:,2),Y);
h = gca;
% cxlim = h.XLim;
% cylim = h.YLim;
% hold on
% Params = cell2mat(Mdl.DistributionParameters); 
% Mu = Params(2*(1:3)-1,1:2); % Extract the means
% Sigma = zeros(2,2,3);
% for j = 1:3
%     Sigma(:,:,j) = diag(Params(2*j,:)).^2; % Create diagonal covariance matrix
%     xlim = Mu(j,1) + 4*[-1 1]*sqrt(Sigma(1,1,j));
%     ylim = Mu(j,2) + 4*[-1 1]*sqrt(Sigma(2,2,j));
%     f = @(x,y) arrayfun(@(x0,y0) mvnpdf([x0 y0],Mu(j,:),Sigma(:,:,j)),x,y);
%     fcontour(f,[xlim ylim]) % Draw contours for the multivariate normal distributions 
% end
% h.XLim = cxlim;
% h.YLim = cylim;
% title('Naive Bayes Classifier -- Fisher''s Iris Data')
% xlabel('Petal Length (cm)')
% ylabel('Petal Width (cm)')
% legend('setosa','versicolor','virginica')
% hold off  

%% Predict probabilities on new data 
Mdl = fitcnb(X,Y,'ClassNames',{'setosa','versicolor','virginica'},'DistributionNames','normal');  
xMax = max(X);
xMin = min(X);
h = 0.01;
[x1Grid,x2Grid] = meshgrid(xMin(1):h:xMax(1),xMin(2):h:xMax(2)); 
[label,PosteriorRegion,cost] = predict(Mdl,[x1Grid(:),x2Grid(:)]); 
% plot the data
h = scatter(x1Grid(:),x2Grid(:),1,PosteriorRegion);
h.MarkerEdgeAlpha = 0.3;  

% plot the posterior probabilities
hold on
gh = gscatter(X(:,1),X(:,2),Y,'k','dx*');
title 'Iris Petal Measurements and Posterior Probabilities (Gaussian)';
xlabel 'Petal length (cm)';
ylabel 'Petal width (cm)';
axis tight
legend(gh,'Location','Best')
hold off

%% specify prior
classNames = {'setosa','versicolor','virginica'}; 
Mdl = fitcnb(X,Y,'ClassNames',classNames,'Prior','uniform'); 

%% Try a kernel density estimation 
figure()
Mdl = fitcnb(X,Y,'ClassNames',{'setosa','versicolor','virginica'},'DistributionNames','kernel');  

xMax = max(X);
xMin = min(X);
h = 0.01;
[x1Grid,x2Grid] = meshgrid(xMin(1):h:xMax(1),xMin(2):h:xMax(2)); 
[label,PosteriorRegion,cost] = predict(Mdl,[x1Grid(:),x2Grid(:)]); 
% plot the data
h = scatter(x1Grid(:),x2Grid(:),1,PosteriorRegion);
h.MarkerEdgeAlpha = 0.3;  

% plot the posterior probabilities
hold on
gh = gscatter(X(:,1),X(:,2),Y,'k','dx*');
title 'Iris Petal Measurements and Posterior Probabilities (Kernel)';
xlabel 'Petal length (cm)';
ylabel 'Petal width (cm)';
axis tight
legend(gh,'Location','Best')
hold off