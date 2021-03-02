%% A script to get down basics on logistic regression using mnrfit 

%% Load fisher iris dataset
clear
load fisheriris
sp = categorical(species); 

%% Pare down to just 2 species and 2 measurements for visualization

meas2 = meas(:,1:2);  

figure()
gscatter(meas2(:,1),meas2(:,2),sp,'rgb','osd')  
title("Fisher Iris Dataset")
xlabel('Sepal length');
ylabel('Sepal width'); 

%% Now perform logistic regression
close all
[B,dev,stats] = mnrfit(meas2,sp);  
pihat_data = mnrval(B,meas2); % should really understand better what mnrval does?
classes = pihat(:,1) > .5; 

% meshgrid
[x,y] = meshgrid(4:.1:8,2:.1:4.5); 
x = x(:);
y = y(:); 
pihat_mesh = mnrval(B,[x y]); 

figure(); 
subplot(1,2,1)
gscatter(meas2(:,1),meas2(:,2),argmax(pihat_data'))   
legend("Sesota","Versicolor","Virginica")  
title("Logistic Regression Results")
subplot(1,2,2) 
gscatter(x,y,argmax(pihat_mesh'))  
legend("Sesota","Versicolor","Virginica")   
title("Meshgrid of logistic regression results")

%% Now try linear discriminant analysis 
lda = fitcdiscr(meas,sp); 
ldaClass = resubPredict(lda); 

figure()
bad = ~strcmp(ldaClass,species);
hold on;
plot(meas(bad,1), meas(bad,2), 'kx');
hold off;
[x,y] = meshgrid(4:.1:8,2:.1:4.5);
x = x(:);
y = y(:);
j = classify([x y],meas(:,1:2),species);
gscatter(x,y,j,'grb','sod') 

% perform xval
cp = cvpartition(species,'KFold',10);
cvlda = crossval(lda,'CVPartition',cp);
ldaCVErr = kfoldLoss(cvlda); 
disp(ldaCVErr) 

%% Last, try out SVM  
% only look at Versicolor vs Virginica, for some nice comparison of a loose
% boundary 
inds = ~strcmp(species,'setosa'); 
X = meas(inds,3:4);
y = species(inds);  
figure()
gscatter(X(:,1),X(:,2),y) 

SVMModel = fitcsvm(X,y);

sv = SVMModel.SupportVectors; % highlight support vectors (pts in margin)
figure
gscatter(X(:,1),X(:,2),y)
hold on
plot(sv(:,1),sv(:,2),'ko','MarkerSize',10)
legend('versicolor','virginica','Support Vector')
hold off


%% the shit you need to do to get the SVM hyperplane...
M  = 200;
X = [mvnrnd([-1,0],2*eye(2),M); ...
    mvnrnd([2,2],eye(2),M); ...
    mvnrnd([-1,4],eye(2),M)];
y = [ones(M,1); 2*ones(M,1); 3*ones(M,1)];
figure;
plot(X(y==1,1),X(y==1,2),'.','DisplayName','1'); hold on
plot(X(y==2,1),X(y==2,2),'.','DisplayName','2');
plot(X(y==3,1),X(y==3,2),'.','DisplayName','3'); hold off

template = templateSVM( 'Standardize', 1);
ecoc = fitcecoc(X,y,'Learners',template);

idx = 1;
svm = ecoc.BinaryLearners{idx}; 

for i = 1:length(ecoc.BinaryLearners)
    svm = ecoc.BinaryLearners{i};
    % Step 1: Choose hyperplane's x1 values
    x1vals = linspace(min(X(:,1)),max(X(:,1)));
      % Step 2: Solve for ((x2-svm.Mu(2))/svm.Sigma(2)) in terms of x1.
      x2fun =@(x1) -svm.Beta(1)/svm.Beta(2) * ((x1-svm.Mu(1))./svm.Sigma(1)) - svm.Bias/svm.Beta(2);
      % Step 3: Calculate corresponding x2 values
      x2vals = x2fun(x1vals) * svm.Sigma(2) + svm.Mu(2);
      hold on;
      plot(x1vals,x2vals,'--', ...
          'DisplayName',mat2str(ecoc.ClassNames(ecoc.CodingMatrix(:,i)~=0)'));
      hold off;
  end
  legend('show','Location','eastoutside','Orientation','vertical');

