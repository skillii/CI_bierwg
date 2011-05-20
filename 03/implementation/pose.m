clear all;
close all;

load('faces.mat');

% 1... Target = individuals;
% 2... Target = direction;
% 3... Target = emotions;
% 4... Target = sunglasses;
targetIndex = 2; 

targetPose1 = full(ind2vec(target2(:, targetIndex)));


[input1, ps] = mapstd(input2');

[input2] = mapstd('apply', input2', ps);

% Set the number of hidden neurons
nHiddenNeurons = 6;
nOutputNeurons = 4;

%create a new feed-forward Network
net = newff(minmax(input1), [nHiddenNeurons, nOutputNeurons], {'logsig', 'logsig'}, 'trainscg');

net.performFcn = 'mse';
net.trainParam.epochs = 300;
net.trainParam.show = 2;

net = init(net);

% Train the network
% Here we have to supply the training input and target data (x_train, t_train).

[net,tr_2hu] = train(net, input1, targetPose1);


%calculate output of learned network, supplied with training data
class_learned = sim(net, input1);
learnedT = hardlim(class_learned - 0.5);



% Calculate confusion matrix on the test set
[error_mat,rate] = confmat(learnedT', targetPose1');

fprintf('Classification rate : %f\n', rate(1));

% % plot first image of test set
figure;

image = mapstd('reverse', input2(:, 1), ps);
image = uint8(reshape(image, 30, 32));
    
imshow(image);


% plot weights of all hidden neurons

figure;
for neuron = 1:nHiddenNeurons
    subplot(3,2,neuron);
    hiddenW1 = reshape(net.IW{1}(neuron, :), 30, 32);
    imagesc(hiddenW1); 
    colormap(gray);
    title(['weights for hidden neuron ' num2str(neuron)]);
end

