%1.1.1

clc;
close all;
clear;



data = importdata('datasets/linearregression_homework.mat');
x_test = data.x_test;
x_train = data.x_train;
y_target = data.y_target;
y_test = data.y_test;
y_train = data.y_train;

l_max = 18;

fig1=figure();



y_our_train = zeros(length(x_train), l_max);

for l = 0:l_max
    disp(['l=', num2str(l)]);
    potenz = ones(size(x_train))*[0:l];

    x_train_mat = [x_train]*ones(1,l+1);
    X = x_train_mat.^potenz;
    W = inv(X'*X)*X'*y_train;
    
    y_our_train(:,l+1) = X*W;

    potenz = ones(size(x_test))*[0:l];
    x_test_mat = [x_test]*ones(1,l+1);
    X_test = x_test_mat.^potenz;

    y(:,l+1) = X_test*W;
    my_plot = plot(x_test, y(:,l+1));
    hold on;
    set(my_plot, 'Color', [l/(l_max+1), l/(l_max+1) , 1-l/(l_max+1)]);
    

end

plot(x_test, y_target, 'b-');
plot(x_train, y_train, ' +');


%das Ganze nocheinmal in skalierter Form geplottet:

fig2=figure();



for l = 0:l_max
   
    my_plot2 = plot(x_test, y(:,l+1));
    hold on;
    set(my_plot2, 'Color', [1, l/(l_max+1) , 1-l/(l_max+1)]);
    axis([-1 1 -4 10]);
end

plot(x_test, y_target, 'b-');
plot(x_train, y_train, ' +');




%plot of basis functions
figure
l=18

x_basis = linspace(-1,1,100)';

potenz = ones(size(x_basis))*[0:l];
x_test_mat = [x_basis]*ones(1,l+1);
X_test = x_test_mat.^potenz;


for i = 1:l
    mask = zeros(l+1, 1);
    mask(i) = 1;

    
    %W2 = W.*mask
    W2 = mask;
    

    
    y_basis = X_test*W2;
    
    my_plot3 = plot(x_basis, y_basis);
    
    hold on;
    set(my_plot3, 'Color', [1-i/(l+1), i/(l+1) , 1-i/(l+1)]);
    axis([-1 1 -1 1]);
    
end




%letzter punkt:

for l = 0:l_max
    mse_test(l+1) = (y(:,l+1) - y_test)' * (y(:,l+1) - y_test) / length(y_test);
    mse_train(l+1) = (y_our_train(:,l+1) - y_train)' * (y_our_train(:,l+1) - y_train) / length(y_our_train);
end

figure

plot(0:l_max, mse_test, 'r-');
hold on;
plot(0:l_max, mse_train, 'b-');
