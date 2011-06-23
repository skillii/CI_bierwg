close all;
clc;


load('data_5_1.mat');



%1.) Scatter Plot:
figure;
plot(LOS_train(1,:), LOS_train(2,:), 'r o');
hold on;
plot(NLOS_train(1,:), NLOS_train(2,:), 'b o');

legend('LOS', 'NLOS');
title('Scatter Plot of training data');

%%

% 2.) Max Likelyhood estimation
x1NLOS = NLOS_train(1,:);
ray_sigma = sqrt(sum(x1NLOS.^2)/(2*length(x1NLOS)));

x1LOS = LOS_train(1,:);
[rice_pdf, x_rice] = ksdensity(x1LOS);


x_ray = linspace(-1, 5, 50);
ray_pdf = x_ray ./ (ray_sigma.^2) .* exp(-x_ray.^2./(2*ray_sigma.^2));


%%

% 3.)

[Nrice,Xrice] = hist(LOS_train(1,:), 100);
[Nray,Xray] = hist(NLOS_train(1,:), 100);

%Nrice = Nrice ./ (sum(Nrice)*(Xrice(end)-Xrice(1)));
%Nray = Nray ./ (sum(Nray)*(Xray(end)-Xray(1)));

figure;
bar(Xrice, Nrice, 'r');
hold on;
bar(Xray, Nray, 'b');

%%