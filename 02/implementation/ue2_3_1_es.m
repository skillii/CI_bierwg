clear all;
close all;

% Ex2 Task 3.1: Simple Regression with Neural net1works, early stopping

linreg_data = open('linearregression_homework.mat');

x_train = linreg_data.x_train';
x_test = linreg_data.x_test';
y_train = linreg_data.y_train';
y_test = linreg_data.y_test';
y_target = linreg_data.y_target';

[x_train_n, ps] = mapstd(x_train);
[x_test_n] = mapstd('apply', x_test, ps);

neurons = 40;

x_train_struct = [x_train_n, x_test_n];
y_train_struct = [y_train, y_test];

% train a neural network

net1 = newff(minmax(x_train_n), [neurons, 1], {'logsig', 'purelin'}, 'trainscg');
    
net1 = init(net1);

net1.performFcn = 'mse';
net1.trainParam.epochs = 700;
net1.trainParam.show = 5;
net1.divideFcn = 'divideind';
net1.divideParam.trainInd = 1:length(x_train_n);
net1.divideParam.valInd = [];
net1.divideParam.testInd = (length(x_train_n) + 1):length(x_train_struct);

[net1, perf1] = train(net1, x_train_struct, y_train_struct);

y_learned_fully = sim(net1, x_test_n);

mse_train1 = sum((y_train - sim(net1, x_train_n)).^2) / length(x_train_n);
mse_test1 = sum((y_test - sim(net1, x_test_n)).^2) / length(x_test_n);

fprintf('fully trained NN with %d neurons: %f mse on training set, %f mse on testset\n', neurons, mse_train1, mse_test1);


% find epoch where mse is minimal

[mse_min mse_min_index] = min(perf1.tperf);
epoch_min = perf1.epoch(mse_min_index);

fprintf('minimal mse is %f at epoch %d\n', mse_min, epoch_min);


% train another NN with epoch_min epochs

%net2 = newff(minmax(x_train_n), [neurons, 1], {'logsig', 'purelin'}, 'trainscg');    
%net2 = init(net2);

net2 = revert(net1);

net2.performFcn = 'mse';
net2.trainParam.epochs = epoch_min;
net2.trainParam.show = 5;
net2.divideFcn = 'divideind';
net2.divideParam.trainInd = 1:length(x_train_n);
net2.divideParam.valInd = [];
net2.divideParam.testInd = (length(x_train_n) + 1):length(x_train_struct);

[net2, perf2] = train(net2, x_train_struct, y_train_struct);

y_learned_es = sim(net2, x_test_n);

mse_train2 = sum((y_train - sim(net2, x_train_n)).^2) / length(x_train_n);
mse_test2 = sum((y_test - sim(net2, x_test_n)).^2) / length(x_test_n);

fprintf('early stopping trained NN with %d neurons: %f mse on training set, %f mse on testset\n', neurons, mse_train2, mse_test2);


% plot learned function for fully trained nn and early stopped nn

figure;
plot(x_test, y_target, 'k--', x_test, y_learned_fully, 'b-', x_test, y_learned_es, 'r-');
title('Learned functions');
xlabel('x');
ylabel('y');
legend('Target Function', 'Fully trained NN', 'NN with Early Stopping', 'Location', 'SouthEast');
