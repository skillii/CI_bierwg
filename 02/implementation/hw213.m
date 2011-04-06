close all;
clear all;
clc;


w0 = [2, 0.5];
eta = [0.2, 0.15, 0.1, 0.05];
alpha = 0.5;

figure(1);
figure(2);

for i = 1:length(eta)
    [weight, error] = gradientDescentAdaptiveHw2(w0, 100, eta(i), alpha);
    plotErrorFunction(weight);
    
    figure(1);
    plot(error);
    hold on;
    
end


%%2.1.3.3

w0 = [-0.2, -0.5];
eta = [0.2, 0.15, 0.1, 0.05];


figure(2);

for i = 1:length(eta)
    [weight, error] = gradientDescentAdaptiveHw2(w0, 100, eta(i), alpha);
    plotErrorFunction(weight);
    
    figure(2);
    plot(error);
    hold on;
    
end
