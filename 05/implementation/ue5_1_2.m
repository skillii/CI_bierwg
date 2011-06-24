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


%Estimation of x2:
phaseLOSmy = sum(LOS_train(2,:)) / length(LOS_train(2,:));
phaseLOSsigmasq = sum((LOS_train(2,:)-phaseLOSmy).^2)/length(LOS_train(2,:));

phaseLOSfunc = @(x) 1/(sqrt(2*pi*phaseLOSsigmasq)) .* exp(-(x-phaseLOSmy).^2./(2*phaseLOSsigmasq));

phaseNLOSfunc = 1/(2*pi);


%%

% 3.)

[Nlos,Xlos] = hist(LOS_train(2,:), 100);
[Nnlos,Xnlos] = hist(NLOS_train(2,:), 100);

Nlos = Nlos ./ (sum(Nlos)*(Xlos(end)-Xlos(1)))*100;
Nnlos = Nnlos ./ (sum(Nnlos)*(Xnlos(end)-Xnlos(1)))*100;



figure;
bar(Xlos, Nlos, 'r');
hold on;
bar(Xnlos, Nnlos, 'b');

x_los = linspace(-pi,pi,100);
x_nlos = linspace(-pi,pi,100);

los_pdf = phaseLOSfunc(x_los);
nlos_pdf = phaseNLOSfunc*ones(size(x_nlos));

plot(x_los, los_pdf, 'r- ', 'LineWidth', 3);
plot(x_nlos, nlos_pdf, 'b- ', 'LineWidth', 3);

xlim([-pi pi]);
ylim([0 1.8]);

legend('LOS', 'NLOS');
title('Estimation of PDF for Phase');

%%

% 4.)

LOSPrior = length(LOS_train(1,:)) / (length(LOS_train(1,:)) + length(NLOS_train(1,:)));
NLOSPrior = length(NLOS_train(1,:)) / (length(LOS_train(1,:)) + length(NLOS_train(1,:)));

%ML Classification:

% first we try to classify the LOS_test data
pNLOS = ray_pdf_func(LOS_test(1,:)) * phaseNLOSfunc;
pLOS = interp1(x_rice, rice_pdf, LOS_test(1,:)) .* phaseLOSfunc(LOS_test(2,:));

figure;
plot(LOS_test(1,pLOS>pNLOS), LOS_test(2,pLOS>pNLOS), 'r o');
hold on;
plot(LOS_test(1,pLOS<=pNLOS), LOS_test(2,pLOS<=pNLOS), 'b o');


correct_classified = sum(pLOS>pNLOS);

%----
% now we try to classify the NLOS_test data
pNLOS = ray_pdf_func(NLOS_test(1,:)) * phaseNLOSfunc;
pLOS = interp1(x_rice, rice_pdf, NLOS_test(1,:)) .* phaseLOSfunc(NLOS_test(2,:));

%figure;
plot(NLOS_test(1,pLOS>pNLOS), NLOS_test(2,pLOS>pNLOS), 'r o');
plot(NLOS_test(1,pLOS<=pNLOS), NLOS_test(2,pLOS<=pNLOS), 'b o');


legend('as LOS classified', 'as NLOS classified');
title('ML-Classification(Amplitude and Phase): Scatter Plot of test data');
xlabel('x1');
ylabel('x2');

correct_classified = correct_classified + sum(pLOS<=pNLOS);

disp(['ML: Es wurden ', num2str(correct_classified/(length(LOS_train(1,:)) + length(NLOS_train(1,:)))*100), '% korrekt klassifiziert']);
%%
%Bayes Classification---------------------------------------------
% first we try to classify the LOS_test data

pNLOS = ray_pdf_func(LOS_test(1,:)) * NLOSPrior * phaseNLOSfunc;
pLOS = interp1(x_rice, rice_pdf, LOS_test(1,:))*LOSPrior .* phaseLOSfunc(LOS_test(2,:));

figure;
plot(LOS_test(1,pLOS>pNLOS), LOS_test(2,pLOS>pNLOS), 'r o');
hold on;
plot(LOS_test(1,pLOS<=pNLOS), LOS_test(2,pLOS<=pNLOS), 'b o');


correct_classified = sum(pLOS>pNLOS);
%----
% now we try to classify the NLOS_test data
pNLOS = ray_pdf_func(NLOS_test(1,:)) * NLOSPrior * phaseNLOSfunc;
pLOS = interp1(x_rice, rice_pdf, NLOS_test(1,:))*LOSPrior  .* phaseLOSfunc(NLOS_test(2,:));

%figure;
plot(NLOS_test(1,pLOS>pNLOS), NLOS_test(2,pLOS>pNLOS), 'r o');
plot(NLOS_test(1,pLOS<=pNLOS), NLOS_test(2,pLOS<=pNLOS), 'b o');


legend('as LOS classified', 'as NLOS classified');
title('Bayes-Classification(Amplitude and Phase): Scatter Plot of test data');
xlabel('x1');
ylabel('x2');

correct_classified = correct_classified + sum(pLOS<=pNLOS);
disp(['Bayes: Es wurden ', num2str(correct_classified/(length(LOS_train(1,:)) + length(NLOS_train(1,:)))*100), '% korrekt klassifiziert']);

