%1.1.3

clc;
close all;
clear;



data = importdata('datasets/linearregression_homework.mat');
x_test = data.x_test;
x_train = data.x_train;
y_target = data.y_target;
y_test = data.y_test;
y_train = data.y_train;

d_max = 18;

fig1=figure();



y_our_train = zeros(length(x_train), d_max);
y = zeros(length(x_test), d_max);

for d = 2:d_max
   
    myh = ones(size(x_train))*linspace(-1, 1, d);
    
    x_train_mat = [x_train]*ones(1,d);
    X = exp((-(x_train_mat-myh).^2)./((2/d).^2));
    X_r = x_train_mat.*exp((-(x_train_mat-myh).^2)./((2/d).^2));
    X = [X, X_r];
    W = inv(X'*X)*X'*y_train;
    
    y_our_train(:,d) = X*W;

    
    
    myh = ones(size(x_test))*linspace(-1, 1, d);
    x_test_mat = [x_test]*ones(1,d);
    X_test = exp((-(x_test_mat-myh).^2)./((2/d).^2));
    X_test_r = x_test_mat.*exp((-(x_test_mat-myh).^2)./((2/d).^2));
    X_test = [X_test, X_test_r];
    
    y(:,d) = X_test*W;
    my_plot = plot(x_test, y(:,d));
    hold on;
    set(my_plot, 'Color', [1, d/(d_max+1) , 1-d/(d_max+1)]);
    

end

plot(x_test, y_target, 'b-');
plot(x_train, y_train, ' +');


%plot of basis functions

x_basis = linspace(-1,1,100)';
d = d_max;
myh = ones(size(x_basis))*linspace(-1, 1, d);

x_basis_mat = [x_basis]*ones(1,d);
X_basis = exp((-(x_basis_mat-myh).^2)./((2/d).^2));

for d = [6 12 18]
    myh = ones(size(x_basis))*linspace(-1, 1, d);

    x_basis_mat = [x_basis]*ones(1,d);
    X_basis = exp((-(x_basis_mat-myh).^2)./((2/d).^2));
    X_basis_r = x_basis_mat.*exp((-(x_basis_mat-myh).^2)./((2/d).^2));
    X_basis = [X_basis, X_basis_r];

    figure
    for k = 2:(d*2)+1
        mask = zeros(d*2, 1);
        mask(k-1) = 1;
    
        y_basis = X_basis*mask;
    
        plot(x_basis, y_basis);
        axis([-1 1 -1 1]);
        hold on;
    end
end




%letzter punkt:

for d = 2:d_max
    mse_test(d) = (y(:,d) - y_test)' * (y(:,d) - y_test) / length(y_test);
    mse_train(d) = (y_our_train(:,d) - y_train)' * (y_our_train(:,d) - y_train) / length(y_our_train);
end

figure

plot(2:d_max, mse_test(2:d_max), 'r-');
hold on;
plot(2:d_max, mse_train(2:d_max), 'b-');
