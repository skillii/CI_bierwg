clear all;
close all;

% Ex2 Task 3: Simple Regression with Neural Networks

linreg_data = open('linearregression_homework.mat');

x_train = linreg_data.x_train';
x_test = linreg_data.x_test';
y_train = linreg_data.y_train';
y_test = linreg_data.y_test';
y_target = linreg_data.y_target';

[x_train_n, ps] = mapstd(x_train);
[x_test_n] = mapstd('apply', x_test, ps);

neurons = [1, 2, 3, 4, 6, 8, 12, 20, 40];

mse_train_vect = zeros(size(neurons));
mse_test_vect = zeros(size(neurons));

net = cell(size(neurons));
perf = cell(size(neurons));

x_train_struct = [x_train_n, x_test_n];
y_train_struct = [y_train, y_test];

for index = 1:length(neurons)
    
    net{index} = newff(minmax(x_train_n), [neurons(index), 1], {'logsig', 'purelin'}, 'trainscg');
    
    net{index} = init(net{index});

    net{index}.performFcn = 'mse';
    net{index}.trainParam.epochs = 700;
    net{index}.trainParam.show = 5;
    net{index}.divideFcn = 'divideind';
    net{index}.divideParam.trainInd = 1:length(x_train_n);
    net{index}.divideParam.valInd = [];
    net{index}.divideParam.testInd = (length(x_train_n) + 1):length(x_train_struct);

    [net{index}, perf{index}] = train(net{index}, x_train_struct, y_train_struct);


    mse_train_vect(index) = sum((y_train - sim(net{index}, x_train_n)).^2) / length(x_train_n);
    mse_test_vect(index) = sum((y_test - sim(net{index}, x_test_n)).^2) / length(x_test_n);

    
    fprintf('%d Neurons: %f mse on training set, %f mse on testset\n', neurons(index), mse_train_vect(index), mse_test_vect(index));
    
end


% plot mse for train and test set

figure;
plot(neurons, mse_train_vect);
title('MSE of training set for defined number of neurons');
xlabel('Number of neurons');
ylabel('MSE');

figure;
plot(neurons, mse_test_vect);
title('MSE of test set for defined number of neurons');
xlabel('Number of neurons');
ylabel('MSE');


% plot performance (mse) on train and test set for 2, 8, 40 neurons

figure;
semilogy(perf{2}.epoch, perf{2}.perf, 'b-', perf{2}.epoch, perf{2}.tperf, 'g-');
title('MSE during training for 2 Neurons');
xlabel('Epoch');
ylabel('MSE');
legend('Training data', 'Test data');

figure;
semilogy(perf{6}.epoch, perf{6}.perf, 'b-', perf{6}.epoch, perf{6}.tperf, 'g-');
title('MSE during training for 8 Neurons');
xlabel('Epoch');
ylabel('MSE');
legend('Training data', 'Test data');

figure;
semilogy(perf{9}.epoch, perf{9}.perf, 'b-', perf{9}.epoch, perf{9}.tperf, 'g-');
title('MSE during training for 40 Neurons');
xlabel('Epoch');
ylabel('MSE');
legend('Training data', 'Test data');


% plot learned function for 2, 8, 40 neurons

%x = linspace(floor(min(x_test)), ceil(max(x_test)), 100);

y_learned_2 = sim(net{2}, x_test_n);
y_learned_8 = sim(net{6}, x_test_n);
y_learned_40 = sim(net{9}, x_test_n);

figure;
plot(x_test, y_target, 'k--', x_test, y_learned_2, 'b-', x_test, y_learned_8, 'g-', x_test, y_learned_40, 'r-');
title('Learned Functions for [2 8 40] Neurons');
xlabel('x');
ylabel('y');
legend('Target Function', '2 Neurons', '8 Neurons', '40 Neurons', 'Location', 'SouthEast');
