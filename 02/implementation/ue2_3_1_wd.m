clear all;
close all;

% Ex2 Task 3.1: Simple Regression with Neural Networks, weight decay

linreg_data = open('linearregression_homework.mat');

x_train = linreg_data.x_train';
x_test = linreg_data.x_test';
y_train = linreg_data.y_train';
y_test = linreg_data.y_test';
y_target = linreg_data.y_target';

[x_train_n, ps] = mapstd(x_train);
[x_test_n] = mapstd('apply', x_test, ps);

neurons = 40;
alpha = [0.9, 0.95, 0.975, 0.99, 0.995, 1.0];

mse_train_vect = zeros(size(alpha));
mse_test_vect = zeros(size(alpha));

y_learned = zeros(size(alpha, 1), size(x_test, 2));

x_train_struct = [x_train_n, x_test_n];
y_train_struct = [y_train, y_test];

for index = 1:length(alpha)
    
    net = newff(minmax(x_train_n), [neurons, 1], {'logsig', 'purelin'}, 'trainscg');
    
    net = init(net);

    net.performFcn = 'msereg';
    net.performParam.ratio = alpha(index);
    net.trainParam.epochs = 700;
    net.trainParam.show = 5;
    net.divideFcn = 'divideind';
    net.divideParam.trainInd = 1:length(x_train_n);
    net.divideParam.valInd = [];
    net.divideParam.testInd = (length(x_train_n) + 1):length(x_train_struct);

    [net, perf] = train(net, x_train_struct, y_train_struct);


    mse_train_vect(index) = sum((y_train - sim(net, x_train_n)).^2) / length(x_train_n);
    mse_test_vect(index) = sum((y_test - sim(net, x_test_n)).^2) / length(x_test_n);
    
    fprintf('%d Neurons, alpha=%f: %f mse on training set, %f mse on testset\n', neurons, alpha(index), mse_train_vect(index), mse_test_vect(index));
    
    y_learned(index, :) = sim(net, x_test_n);
    
end


% plot mse for train and test set

figure;
plot(alpha, mse_train_vect);
title('MSE of training set for defined regularization factor alpha');
xlabel('alpha');
ylabel('MSE');

figure;
plot(alpha, mse_test_vect);
title('MSE of test set for defined regularization factor alpha');
xlabel('alpha');
ylabel('MSE');


% plot learned function for lowest, highest and best value of alpha

[mse_min min_index] = min(mse_test_vect);
best_alpha = alpha(min_index);

disp(['minimum mse is ' num2str(mse_min) ' with alpha=' num2str(best_alpha)]);

figure;
plot(x_test, y_target, 'k--', x_test, y_learned(1, :), 'b-', x_test, y_learned(end, :), 'g-', x_test, y_learned(min_index, :), 'r-');
title('Learned Functions for lowest/highest/best alpha');
xlabel('x');
ylabel('y');
legend('Target Function', ['lowest alpha (' num2str(alpha(1)) ')'], ['highest alpha (' num2str(alpha(end)) ')'], ['best alpha (' num2str(alpha(min_index)) ')'], 'Location', 'SouthEast');
