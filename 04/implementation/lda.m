% CI HW4 Task 2: LDA
close all;
clear all;
clc;

dataset = load('data_homework.mat');



%% perform basic LDA

% calculate mean values for classes and weight vector
m1_ = mean(dataset.X_1, 2);
m2_ = mean(dataset.X_2, 2);

w = m2_ - m1_;

m1 = w' * m1_;
m2 = w' * m2_;


% plot the data points
figure;
plot3(dataset.X_1(1,:), dataset.X_1(2,:), dataset.X_1(3,:), 'g.', dataset.X_2(1,:), dataset.X_2(2,:), dataset.X_2(3,:), 'r.');
hold on;

plot3([m1_(1) m2_(1)], [m1_(2) m2_(2)], [m1_(3) m2_(3)], 'b-', 'LineWidth', 3);
title('Data Points for LDA');
xlabel('x');
ylabel('y');
zlabel('z');
legend('Class 1', 'Class 2', 'Weight Vector for LDA');

% transform to 1dimensional space and plot
x1 = w' * dataset.X_1;
x2 = w' * dataset.X_2;

% plot data points
figure;
plot(x1, 0, 'gx');
hold on;
plot(x2, 0, 'rx');
title('Data Points transformed to 1dimensional Space with LDA');
xlabel('$\hat{x}$', 'interpreter', 'latex');
legend('Class 1', 'Class 2');

% plot histograms
figure;
hist(x1);
hold on;
hist(x2);
h = findobj(gca, 'Type', 'patch');
set(h(1), 'FaceColor', 'g');
set(h(2), 'FaceColor', 'r');
title('Histograms for transformed Data Points with LDA');
xlabel('$\hat{x}$', 'interpreter', 'latex');
ylabel('# of Elements');
legend('Class 1', 'Class 2');

% find treshold and perform classification
[bins, xout] = hist([x1 x2]);
[~, peak_pos] = findpeaks(bins);
[~, treshold_index] = min(bins(peak_pos(1):peak_pos(end)));
treshold = xout(treshold_index + peak_pos(1) - 1);

cperf1 = sum(x1 < treshold) / numel(x1);
cperf2 = sum(x2 > treshold) / numel(x2);
cperf = (sum(x1 < treshold) + sum(x2 > treshold)) / (numel(x1) + numel(x2));

disp(['LDA: Threshold is ' num2str(treshold) ', Performance on Class 1 is ' num2str(cperf1) ', Performance on Class 2 is ' num2str(cperf2) ', Overall Performance is ' num2str(cperf)]);




%% perform Fisher LDA

% calculate mean values for classes and within separation matrix
m1_ = mean(dataset.X_1, 2);
m2_ = mean(dataset.X_2, 2);

M1_ = repmat(m1_, 1, size(dataset.X_1, 2));
M2_ = repmat(m2_, 1, size(dataset.X_2, 2));

S1 = (dataset.X_1 - M1_) * (dataset.X_1 - M1_)';
S2 = (dataset.X_2 - M2_) * (dataset.X_2 - M2_)';
Sw = S1 + S2;

w = Sw \ (m2_ - m1_);

% plot the data points
figure;
plot3(dataset.X_1(1,:), dataset.X_1(2,:), dataset.X_1(3,:), 'g.', dataset.X_2(1,:), dataset.X_2(2,:), dataset.X_2(3,:), 'r.');
hold on;

scale = 80;

plot3([m1_(1) m1_(1)+scale*w(1)], [m1_(2) m1_(2)+scale*w(2)], [m1_(3) m1_(3)+scale*w(3)], 'b-', 'LineWidth', 3);
title('Data Points for Fisher-LDA');
xlabel('x');
ylabel('y');
zlabel('z');
legend('Class 1', 'Class 2', 'Weight Vector for Fisher-LDA');

% transform to 1dimensional space and plot
x1 = w' * dataset.X_1;
x2 = w' * dataset.X_2;

% plot data points
figure;
plot(x1, 0, 'gx');
hold on;
plot(x2, 0, 'rx');
title('Data Points transformed to 1dimensional Space with Fisher-LDA');
xlabel('$\hat{x}$', 'interpreter', 'latex');
legend('Class 1', 'Class 2');

% plot histograms
figure;
hist(x1);
hold on;
hist(x2);
h = findobj(gca, 'Type', 'patch');
set(h(1), 'FaceColor', 'g');
set(h(2), 'FaceColor', 'r');
title('Histograms for transformed Data Points with Fisher-LDA');
xlabel('$\hat{x}$', 'interpreter', 'latex');
ylabel('# of Elements');
legend('Class 1', 'Class 2');

% find treshold and perform classification
[bins, xout] = hist([x1 x2]);
[~, peak_pos] = findpeaks(bins);
[~, treshold_index] = min(bins(peak_pos(1):peak_pos(end)));
treshold = xout(treshold_index + peak_pos(1) - 1);

cperf1 = sum(x1 < treshold) / numel(x1);
cperf2 = sum(x2 > treshold) / numel(x2);
cperf = (sum(x1 < treshold) + sum(x2 > treshold)) / (numel(x1) + numel(x2));

disp(['Fisher-LDA: Threshold is ' num2str(treshold) ', Performance on Class 1 is ' num2str(cperf1) ', Performance on Class 2 is ' num2str(cperf2) ', Overall Performance is ' num2str(cperf)]);
