clear all;
close all;

load('faces.mat');

% 1... Target = individuals;
% 2... Target = direction;
% 3... Target = emotions;
% 4... Target = sunglasses;
targetIndex = 1; 

target1 = full(ind2vec(target1(:, targetIndex)));
target2 = full(ind2vec(target2(:, targetIndex)));

[input1, ps] = mapstd(input1');
[input2] = mapstd('apply', input2', ps);

nTimes = 10;
net = cell(nTimes,1);
perf = cell(nTimes,1);

% Set the number of hidden neurons
nHiddenNeurons = 20;
Test.T = target2;
Test.P = input2;

defaultnet = newff(minmax(input1), [nHiddenNeurons, 20], {'logsig', 'logsig'}, 'trainscg');

defaultnet.performFcn = 'mse';
defaultnet.trainParam.epochs = 1000;
defaultnet.trainParam.show = 2;

mse_train_vect = zeros(nTimes,1);
mse_test_vect = zeros(nTimes,1);

for k = 1:nTimes
    net{k} = init(defaultnet);

    % Train the network
    % Here we have to supply the training input and target data (x_train, t_train).

    [net{k},tr_2hu] = train(net{k}, input1, target1, [],[],[], Test);

    mse_train_vect(k) = tr_2hu.perf(end); %sum((y_train - sim(net{index}, x_train_n)).^2) / length(x_train_n);
    mse_test_vect(k) = tr_2hu.tperf(end); %sum((y_test - sim(net{index}, x_test_n)).^2) / length(x_test_n);

end


figure
hist(mse_train_vect,10)
xlabel('MSE');
<<<<<<< HEAD
ylabel('count');
=======
ylabel('iteration');
>>>>>>> 90b6d64d9e1e825bdee8ad3ac8cdcdf4f04c021f
title('MSE on train-data');

figure
hist(mse_test_vect,10)
xlabel('MSE');
<<<<<<< HEAD
ylabel('count');
=======
ylabel('iteration');
>>>>>>> 90b6d64d9e1e825bdee8ad3ac8cdcdf4f04c021f
title('MSE on test-data');

[~, best_net] = min(mse_test_vect);


%%

% Calculate the mean classification error on the training set:

class_learned = sim(net{best_net}, input2);

realT = target2;
learnedT = hardlim(class_learned - 0.5);

false_classified_log = sum(realT-learnedT,1) ~= 0;
false_classified_images = input2(:,false_classified_log);
right_classified_log = sum(realT-learnedT,1) == 0;
right_classified_images = input2(:,right_classified_log);

% error = sum(vec2ind(realT) ~= vec2ind(learnedT)) / size(realT, 2);
% 
% fprintf('Mean classification error on training set: %f\n', error);


% Calculate confusion matrix on the test set
[error_mat,rate] = confmat(learnedT', target2');

fprintf('Classification rate : %f\n', rate(1));

numfalseplots = min([10, sum(false_classified_log)]);
figure;

sum(false_classified_log)
% % plot first image of test set
for k = 1:numfalseplots
subplot(4,5, k);
image = mapstd('reverse', false_classified_images(:,k), ps);
image = uint8(reshape(image, 30, 32));
    
imshow(image);
title('false classified');
end


numrightplots = min([10, sum(right_classified_log)]);


% % plot first image of test set
for k = 1:numrightplots
subplot(4,5, k+10);
image = mapstd('reverse', right_classified_images(:,k), ps);
image = uint8(reshape(image, 30, 32));
    
imshow(image);
title('right classified');
end

% plot weights of all hidden neurons
