% Homework 6: k-means

%clc;
%clear all;
%close all;


dataset = load('vowels.mat');

M = 5;
%mu_0 = repmat(min(dataset.allvow), 5, 1) + repmat(max(dataset.allvow) - min(dataset.allvow), 5, 1) .* rand(5,2);
mu_0 = 1.0e+03*[0.4, 1; 0.6, 1.2; 0.5, 1.4; 0.7, 1.6; 0.8, 1.8];

max_iter = 50;


% generate scatter plots

 figure;
 plot(dataset.a(:,1), dataset.a(:,2), 'b.');
 hold on;
 plot(dataset.e(:,1), dataset.e(:,2), 'g.');
 plot(dataset.i(:,1), dataset.i(:,2), 'r.');
 plot(dataset.o(:,1), dataset.o(:,2), 'k.');
 plot(dataset.y(:,1), dataset.y(:,2), 'm.');
 xlabel('x1');
 ylabel('x2');
 title('Scatter Plot of Labeled Input Data');
 legend('A', 'E', 'I', 'O', 'U', 'Y');

%%
% run k-means algorithm

[mu, D] = k_means(dataset.allvow, M, mu_0, max_iter);


% plot GMM
%%
for m = 1:M
    plotGaussContour(mu(m,:), 100);
end


% plot log-Likelihood

% figure;
% plot(L);
% xlabel('Iteration');
% ylabel('log-Likelihood');
% title('log-Likelihood Behaviour');
