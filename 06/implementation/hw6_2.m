% Homework 6: k-means

%clc;
%clear all;
%close all;


dataset = load('vowels.mat');

M = 5;
mu_0 = repmat(min(dataset.allvow), 5, 1) + repmat(max(dataset.allvow) - min(dataset.allvow), 5, 1) .* rand(5,2);
%mu_0 = 1.0e+03*[0.4, 1; 0.6, 1.2; 0.5, 1.4; 0.7, 1.6; 0.8, 1.8];

max_iter = 1000;


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

[mu, D, ind] = k_means(dataset.allvow, M, mu_0, max_iter);


% plot GMM
%%
for m = 1:M
    %plotGaussContour(mu(m,:), 100);
    hold on;
    plot(mu(m,1), mu(m,2),'c o', 'Linewidth', 10); 
end

%%
figure;
 plot(dataset.allvow(ind == 1,1), dataset.allvow(ind == 1,2), 'b.');
 hold on;
 plot(dataset.allvow(ind == 2,1), dataset.allvow(ind == 2,2), 'g.');
 plot(dataset.allvow(ind == 3,1), dataset.allvow(ind == 3,2), 'r.');
 plot(dataset.allvow(ind == 4,1), dataset.allvow(ind == 4,2), 'k.');
 plot(dataset.allvow(ind == 5,1), dataset.allvow(ind == 5,2), 'm.');
 xlabel('x1');
 ylabel('x2');
 title('Scatter Plot of classified input data(KMEANS)');
 legend('A', 'E', 'I', 'O', 'U', 'Y');

for m = 1:M
    %plotGaussContour(mu(m,:), 100);
    hold on;
    plot(mu(m,1), mu(m,2),'c o', 'Linewidth', 10); 
end

% plot log-Likelihood

% figure;
% plot(L);
% xlabel('Iteration');
% ylabel('log-Likelihood');
% title('log-Likelihood Behaviour');
