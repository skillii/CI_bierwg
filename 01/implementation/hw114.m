clc;
close all;
clear;



data = importdata('datasets/linearregression_homework.mat');
x_test = data.x_test;
x_train = data.x_train;
y_target = data.y_target;
y_test = data.y_test;
y_train = data.y_train;


%Polynomial Basis function
l = 18;


alpha = logspace(-10,0,200);

y_our_train_poly = zeros(length(x_train), length(alpha));
y_our_test_poly = zeros(length(x_test), length(alpha));


potenz = ones(size(x_train))*[0:l];
x_train_mat = [x_train]*ones(1,l+1);
X_train = x_train_mat.^potenz;

potenz = ones(size(x_test))*[0:l];
x_test_mat = [x_test]*ones(1,l+1);
X_test = x_test_mat.^potenz;


W_mean = zeros(1,length(alpha));

%output y berechnen
for i=1:length(alpha)
    W = inv(X_train'*X_train - alpha(i)^2*length(x_train)*eye(l+1))*X_train'*y_train;
    
    W_mean(i) = mean(W);
    
    y_our_train_poly(:,i) = X_train*W;

    y_our_test_poly(:,i) = X_test*W;
    
end





%MSE plotten:
for i = 1:length(alpha)
    mse_test_poly(i) = (y_our_test_poly(:,i) - y_test)' * (y_our_test_poly(:,i) - y_test) / length(y_test);
    
    mse_train_poly(i) = (y_our_train_poly(:,i) - y_train)' * (y_our_train_poly(:,i) - y_train) / length(y_our_train_poly);
end

figure

semilogx(alpha, mse_test_poly, 'r-');
hold on;
semilogx(alpha, mse_train_poly, 'b-');


figure();
%plot of lowest, highest and best alpha
[Y,I] = min(mse_test_poly);

plot(x_test, y_our_test_poly(:,1), 'r-');
hold on;
plot(x_test, y_our_test_poly(:,length(alpha)), 'g-');
plot(x_test, y_our_test_poly(:,I), 'b-');

plot(x_train, y_train, ' +');

figure

semilogx(alpha, W_mean);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Radial Basis
d = 18;

y_our_train_radial = zeros(length(x_train), d);
y_our_test_radial = zeros(length(x_test), d);


myh_train = ones(size(x_train))*linspace(-1, 1, d);
myh_test = ones(size(x_test))*linspace(-1, 1, d);

x_train_mat = [x_train]*ones(1,d);
X_train = exp((-(x_train_mat-myh_train).^2)./((2/d).^2));
x_test_mat = [x_test]*ones(1,d);
X_test = exp((-(x_test_mat-myh_test).^2)./((2/d).^2));


%output y berechnen
for i=1:length(alpha)
    W = inv(X_train'*X_train - alpha(i)^2*length(x_train)*eye(d))*X_train'*y_train;
    
    W_mean(i) = mean(W);

    y_our_train_radial(:,i) = X_train*W;

    y_our_test_radial(:,i) = X_test*W;
    
end


%MSE plotten:
for i = 1:length(alpha)
    mse_test_radial(i) = (y_our_test_radial(:,i) - y_test)' * (y_our_test_radial(:,i) - y_test) / length(y_test);
    
    mse_train_radial(i) = (y_our_train_radial(:,i) - y_train)' * (y_our_train_radial(:,i) - y_train) / length(y_our_train_radial);
end

figure

semilogx(alpha, mse_test_radial, 'r-');
hold on;
semilogx(alpha, mse_train_radial, 'b-');


figure();
%plot of lowest, highest and best alpha
[Y,I] = min(mse_test_radial);

plot(x_test, y_our_test_radial(:,1), 'r-');
hold on;
plot(x_test, y_our_test_radial(:,length(alpha)), 'g-');
plot(x_test, y_our_test_radial(:,I), 'b-');

plot(x_train, y_train, ' +');


figure

semilogx(alpha, W_mean);

