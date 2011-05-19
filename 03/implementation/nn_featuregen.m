% Homework 4, Task 4: Neural Networks as Feature Generator

clear all;
close all;
%clc;


% generate digits

letter8 = [ 0 0 0 0 0 0 0 0;
            0 0 0 1 1 0 0 0;
            0 0 1 0 0 1 0 0;
            0 0 1 0 0 1 0 0;
            0 0 0 1 1 0 0 0;
            0 0 1 0 0 1 0 0;
            0 0 1 0 0 1 0 0;
            0 0 0 1 1 0 0 0 ];

letter3 = [ 0 0 0 0 0 0 0 0;
            0 0 0 1 1 0 0 0;
            0 0 1 0 0 1 0 0;
            0 0 0 0 0 1 0 0;
            0 0 0 1 1 0 0 0;
            0 0 0 0 0 1 0 0;
            0 0 1 0 0 1 0 0;
            0 0 0 1 1 0 0 0 ];
        
letter0 = [ 0 0 0 0 0 0 0 0;
            0 0 0 1 1 0 0 0;
            0 0 1 0 0 1 0 0;
            0 0 1 0 0 1 0 0;
            0 0 1 0 0 1 0 0;
            0 0 1 0 0 1 0 0;
            0 0 1 0 0 1 0 0;
            0 0 0 1 1 0 0 0 ];


% generate training data from digits by adding noise and target data

nTraining = 100;

train8 = repmat(letter8(:)', nTraining, 1) + (-0.5 + rand(nTraining, length(letter8(:)')));
train3 = repmat(letter3(:)', nTraining, 1) + (-0.5 + rand(nTraining, length(letter3(:)')));
train0 = repmat(letter0(:)', nTraining, 1) + (-0.5 + rand(nTraining, length(letter0(:)')));

trainingSet = [train8; train3; train0];

[trainingSet, ps] = mapstd(trainingSet');


target8 = ones(nTraining, 1) * 1;  % 1 == letter8
target3 = ones(nTraining, 1) * 2;  % 2 == letter3
target0 = ones(nTraining, 1) * 3;  % 3 == letter0

targetSet = [target8; target3; target0];
targetMat = full(ind2vec(targetSet'));


% train a NN

nHiddenNeurons = 2;

net = newff(minmax(trainingSet), [nHiddenNeurons, 3], {'logsig', 'logsig'}, 'trainscg');

net.performFcn = 'mse';
net.trainParam.epochs = 300;

net = init(net);

[net, perf] = train(net, trainingSet, targetMat);


% calculate the mean classification error on the training set

learned = sim(net, trainingSet);

realTarget = targetMat;
learnedTarget = hardlim(learned - 0.5);

error = sum(sum(realTarget ~= learnedTarget)) / size(realTarget, 1);

fprintf('Mean classification error on training set: %f\n', error);


% display input to hidden weights

figure;

for hiddenNeuron = 1:nHiddenNeurons

    hiddenW = reshape(net.IW{1}(hiddenNeuron, :), 8, 8);
    
    subplot(1, nHiddenNeurons, hiddenNeuron);
    imagesc(hiddenW);
    colormap(gray);
    title(['Weights of Hidden Layer Neuron ' num2str(hiddenNeuron)]);
    
end

% display the different patterns to learn

figure;

subplot(1, 3, 1);
imagesc(letter8);
colormap(gray);
title('Pattern for Letter 8');

subplot(1, 3, 2);
imagesc(letter3);
colormap(gray);
title('Pattern for Letter 3');

subplot(1, 3, 3);
imagesc(letter0);
colormap(gray);
title('Pattern for Letter 0');
