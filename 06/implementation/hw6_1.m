% Homework 6: Expectation Maximation

clc;
clear all;
close all;


dataset = load('vowels.mat');

M = 5;
alpha_0 = [1; 1; 1; 1; 1];
mu_0 = repmat(min(dataset.allvow), 5, 1) + repmat(max(dataset.allvow) - min(dataset.allvow), 5, 1) .* rand(5,2);
Sigma_0 = cat(3, [100000, 100; 100, 100000], [100000, 100; 100, 100000], [100000, 100; 100, 100000], [100000, 100; 100, 100000], [100000, 100; 100, 100000]);
max_iter = 1000;


% generate scatter plots

figure;
plot(dataset.a(:,1), dataset.a(:,2), 'b.');
hold on;
plot(dataset.e(:,1), dataset.e(:,2), 'g.');
plot(dataset.i(:,1), dataset.i(:,2), 'r.');
plot(dataset.o(:,1), dataset.o(:,2), 'k.');
%plot(dataset.u(:,1), dataset.u(:,2), 'c.');
plot(dataset.y(:,1), dataset.y(:,2), 'm.');
xlabel('x1');
ylabel('x2');
title('Scatter Plot of Labeled Input Data');
legend('A', 'E', 'I', 'O', 'Y');


% run EM algorithm

[alpha, mu, Sigma, L] = EM(dataset.allvow, M, alpha_0, mu_0, Sigma_0, max_iter);


% plot GMM

for m = 1:M
    plotGaussContour(mu(m,:), Sigma(:,:,m));
end


% plot log-Likelihood

figure;
plot(L);
xlabel('Iteration');
ylabel('log-Likelihood');
title('log-Likelihood Behaviour');
