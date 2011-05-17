clear all;
close all;

load('faces.mat');

% 1... Target = individuals;
% 2... Target = direction;
% 3... Target = emotions;

targetIndex = 4; % 4... Target = sunglasses;

targetSunglasses1 = target1(:, targetIndex) == 2;
targetSunglasses2 = target2(:, targetIndex) == 2;



[input1, ps] = mapstd(input1');
[input2] = mapstd('apply', input2', ps);

% Set the number of hidden neurons
nHiddenNeurons = 4;
net = newff(minmax(input1), [nHiddenNeurons, 1], {'logsig', 'logsig'}, 'trainscg');

net.performFcn = 'mse';
net.trainParam.epochs = 1000;
net.trainParam.show = 2;

net = init(net);

% Train the network
% Here we have to supply the training input and target data (x_train, t_train).

Test.T = targetSunglasses2';
Test.P = input2;

[net,tr_2hu] = train(net, input1, targetSunglasses1', [],[],[], Test);

% Calculate the mean classification error on the training set:

class_learned = sim(net, input1);

realT = targetSunglasses1;
learnedT = hardlim(class_learned' - 0.5);

error = sum(realT ~= learnedT) / size(realT, 1);

fprintf('Mean classification error on training set: %f\n', error);



% Calculate the mean classification error on the testset:
class_learned = sim(net, Test.P);

realT = targetSunglasses2;
learnedT = hardlim(class_learned' - 0.5);

error = sum(realT ~= learnedT) / size(realT, 1);

fprintf('Mean classification error on test set: %f\n', error);

% Calculate confusion matrix on the test set
[error_mat,rate] = confmat(full(ind2vec(learnedT + 1)'), full(ind2vec(Test.T + 1)'));

fprintf('Classification rate : %f\n', rate(1));

% plot first image of test set
figure;

image = mapstd('reverse', input2(:, 1), ps);
image = uint8(reshape(image, 30, 32));
    
imshow(image);

% plot weights of first hidden neuron image of test set
figure;

neuron = 1;
hiddenW1 = reshape(net.IW{1}(neuron, :), 30, 32);
imagesc(hiddenW1); 
colormap(gray);



