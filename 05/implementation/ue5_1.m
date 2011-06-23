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
xlabel('x1');
ylabel('x2');

%%

% 2.) Max Likelyhood estimation
x1NLOS = NLOS_train(1,:);
ray_sigma = sqrt(sum(x1NLOS.^2)/(2*length(x1NLOS)));

x1LOS = LOS_train(1,:);
[rice_pdf, x_rice] = ksdensity(x1LOS);


x_ray = linspace(-1, 5, 50);
ray_pdf_func = @(x) x ./ (ray_sigma.^2) .* exp(-x.^2./(2*ray_sigma.^2));
ray_pdf = x_ray ./ (ray_sigma.^2) .* exp(-x_ray.^2./(2*ray_sigma.^2));


%%

% 3.)

[Nrice,Xrice] = hist(LOS_train(1,:), 100);
[Nray,Xray] = hist(NLOS_train(1,:), 100);

Nrice = Nrice ./ (sum(Nrice)*(Xrice(end)-Xrice(1)))*100;
Nray = Nray ./ (sum(Nray)*(Xray(end)-Xray(1)))*100;



figure;
bar(Xrice, Nrice, 'r');
hold on;
bar(Xray, Nray, 'b');

plot(x_rice, rice_pdf, 'r- ', 'LineWidth', 3);
plot(x_ray, ray_pdf, 'b- ', 'LineWidth', 3);

xlim([0 5]);
ylim([0 1]);

legend('LOS', 'NLOS');
title('Estimation of Rayleigh and Rice PDF');

%%

% 4.)

LOSPrior = length(LOS_train(1,:)) / (length(LOS_train(1,:)) + length(NLOS_train(1,:)));
NLOSPrior = length(NLOS_train(1,:)) / (length(LOS_train(1,:)) + length(NLOS_train(1,:)));

%ML Classification:

% first we try to classify the LOS_test data
pNLOS = ray_pdf_func(LOS_test(1,:));
pLOS = interp1(x_rice, rice_pdf, LOS_test(1,:));

figure;
plot(LOS_test(1,pLOS>pNLOS), LOS_test(2,pLOS>pNLOS), 'r o');
hold on;
plot(LOS_test(1,pLOS<=pNLOS), LOS_test(2,pLOS<=pNLOS), 'b o');


correct_classified = sum(pLOS>pNLOS);

%----
% now we try to classify the NLOS_test data
pNLOS = ray_pdf_func(NLOS_test(1,:));
pLOS = interp1(x_rice, rice_pdf, NLOS_test(1,:));

%figure;
plot(NLOS_test(1,pLOS>pNLOS), NLOS_test(2,pLOS>pNLOS), 'r o');
plot(NLOS_test(1,pLOS<=pNLOS), NLOS_test(2,pLOS<=pNLOS), 'b o');


legend('as LOS classified', 'as NLOS classified');
title('ML-Classification: Scatter Plot of test data');
xlabel('x1');
ylabel('x2');

correct_classified = correct_classified + sum(pLOS<=pNLOS);

disp(['ML: Es wurden ', num2str(correct_classified/(length(LOS_train(1,:)) + length(NLOS_train(1,:)))*100), '% korrekt klassifiziert']);
%%
%Bayes Classification---------------------------------------------
% first we try to classify the LOS_test data

pNLOS = ray_pdf_func(LOS_test(1,:)) * NLOSPrior;
pLOS = interp1(x_rice, rice_pdf, LOS_test(1,:))*LOSPrior;

figure;
plot(LOS_test(1,pLOS>pNLOS), LOS_test(2,pLOS>pNLOS), 'r o');
hold on;
plot(LOS_test(1,pLOS<=pNLOS), LOS_test(2,pLOS<=pNLOS), 'b o');


correct_classified = sum(pLOS>pNLOS);
%----
% now we try to classify the NLOS_test data
pNLOS = ray_pdf_func(NLOS_test(1,:)) * NLOSPrior;
pLOS = interp1(x_rice, rice_pdf, NLOS_test(1,:))*LOSPrior;

%figure;
plot(NLOS_test(1,pLOS>pNLOS), NLOS_test(2,pLOS>pNLOS), 'r o');
plot(NLOS_test(1,pLOS<=pNLOS), NLOS_test(2,pLOS<=pNLOS), 'b o');


legend('as LOS classified', 'as NLOS classified');
title('Bayes-Classification: Scatter Plot of test data');
xlabel('x1');
ylabel('x2');

correct_classified = correct_classified + sum(pLOS<=pNLOS);
disp(['Bayes: Es wurden ', num2str(correct_classified/(length(LOS_train(1,:)) + length(NLOS_train(1,:)))*100), '% korrekt klassifiziert']);

