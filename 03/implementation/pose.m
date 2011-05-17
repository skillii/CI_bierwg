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
net = newff(minmax(input1), [nHiddenNeurons, 4], {'logsig', 'logsig'}, 'trainscg');

net.performFcn = 'mse';
net.trainParam.epochs = 300;
net.trainParam.show = 2;

net = init(net);

% Train the network
% Here we have to supply the training input and target data (x_train, t_train).

[net,tr_2hu] = train(net, input1, targetPose1);

% Calculate the mean classification error on the training set:

class_learned = sim(net, input1);

realT = targetPose1;
learnedT = hardlim(class_learned - 0.5);

% error = sum(vec2ind(realT) ~= vec2ind(learnedT)) / size(realT, 2);
% 
% fprintf('Mean classification error on training set: %f\n', error);



% Calculate confusion matrix on the test set
[error_mat,rate] = confmat(learnedT', targetPose1');

fprintf('Classification rate : %f\n', rate(1));

% % plot first image of test set
figure;

image = mapstd('reverse', input2(:, 1), ps);
image = uint8(reshape(image, 30, 32));
    
imshow(image);

% plot weights of all hidden neurons

for neuron = 1:nHiddenNeurons

    figure;
    hiddenW1 = reshape(net.IW{1}(neuron, :), 30, 32);
    imagesc(hiddenW1); 
    colormap(gray);
    title(['weights for hidden neuron ' num2str(neuron)]);

end

