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
plot(dataset.o(:,1), dataset.o(:,2), 'c.');
plot(dataset.y(:,1), dataset.y(:,2), 'm.');
xlabel('x1');
ylabel('x2');
title('Scatter Plot of Labeled Input Data and Results of EM');
legend('A', 'E', 'I', 'O', 'Y');


% run EM algorithm

[alpha, mu, Sigma, L, r_mn] = EM(dataset.allvow, M, alpha_0, mu_0, Sigma_0, max_iter);


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



%% soft-classification

soft_class = vec2ind(r_mn')';

class1 = dataset.allvow(soft_class == 1, :);
class2 = dataset.allvow(soft_class == 2, :);
class3 = dataset.allvow(soft_class == 3, :);
class4 = dataset.allvow(soft_class == 4, :);
class5 = dataset.allvow(soft_class == 5, :);


% generate scatter plots

figure;
plot(class1(:,1), class1(:,2), 'b.');
hold on;
plot(class2(:,1), class2(:,2), 'g.');
plot(class3(:,1), class3(:,2), 'r.');
plot(class4(:,1), class4(:,2), 'c.');
plot(class5(:,1), class5(:,2), 'm.');
xlabel('x1');
ylabel('x2');
title('Scatter Plot of Soft-Classified Input Data');
legend('Component 1', 'Component 2', 'Component 3', 'Component 4', 'Component 5');



%% and now for something completely different: the same with diagonal Sigma

Sigma_0_diag = repmat([100000 100000], M, 1);


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
title('Scatter Plot of Labeled Input Data and Results of EM with diagonal \Sigma');
legend('A', 'E', 'I', 'O', 'Y');


% run EM algorithm

[alpha_diag, mu_diag, Sigma_diag, L_diag] = EM_diagonal(dataset.allvow, M, alpha_0, mu_0, Sigma_0_diag, max_iter);


% plot GMM

for m = 1:M
    plotGaussContour(mu_diag(m,:), diag(Sigma_diag(m,:)));
end


% plot log-Likelihood

figure;
plot(L_diag);
xlabel('Iteration');
ylabel('log-Likelihood');
title('log-Likelihood Behaviour for diagonal \Sigma');
