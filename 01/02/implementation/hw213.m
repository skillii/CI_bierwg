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
    my_plot = plot(error);
    hold on;
    
    set(my_plot, 'Color', [i/length(eta), i/length(eta) , 1-i/length(eta)]);
    

    
end

% title etc for errorplot 1
title('Evolution of the error vector for w_0 = [2 0.5] (Gradient Descent with Impulse Term and Adaptive Learning Rate)');
xlabel('Iteration');
ylabel('Error');
legend('\eta = 0.2', '\eta = 0.15', '\eta = 0.1', '\eta = 0.05');



%%2.1.3.3

w0 = [-0.2, -0.5];
eta = [0.2, 0.15, 0.1, 0.05];


figure(2);

for i = 1:length(eta)
    [weight, error] = gradientDescentAdaptiveHw2(w0, 100, eta(i), alpha);
    plotErrorFunction(weight);
    
    figure(2);
    my_plot = plot(error);
    hold on;
    
    set(my_plot, 'Color', [i/length(eta), i/length(eta) , 1-i/length(eta)]);
    
end

% title etc for errorplot 2
title('Evolution of the error vector for w_0 = [-0.2 -0.5] (Gradient Descent with Impulse Term and Adaptive Learning Rate)');
xlabel('Iteration');
ylabel('Error');
legend('\eta = 0.2', '\eta = 0.15', '\eta = 0.1', '\eta = 0.05');
