%PCA

clc;
close all;
clear;



data = importdata('datasets_HW4/data_homework.mat');
X = data.X;
X1 = data.X_1;
X2 = data.X_2;

C_x = cov(X');


[EV,EW] = eig(C_x);



plot3(X1(1,:),X1(2,:),X1(3,:), 'r+');
hold on
plot3(X2(1,:), X2(2,:), X2(3,:), 'bo');
grid
xlabel('x')
ylabel('y')
zlabel('z')

Mean_X = mean(X')';


[maxval,index] = max(max(EW))


lineX = [-EV(:,1),EV(:,1)];
line1 = [-EV(:,2),EV(:,2)];
line2 = [-EV(:,3),EV(:,3)];

scale = EW(1,1);
scale1 = EW(2,2);
scale2 = EW(3,3);

line(scale*lineX(1,:), scale*lineX(2,:), scale*lineX(3,:), 'LineWidth', 3, 'Color', [0,1,0]);
line(scale1*line1(1,:), scale1*line1(2,:), scale1*line1(3,:), 'LineWidth', 3, 'Color', [1,0,0]);
line(scale2*line2(1,:), scale2*line2(2,:), scale2*line2(3,:), 'LineWidth', 3, 'Color', [0,0,1]);

Y = EV(:,index)'*X;
Y_1 = EV(:,index)'*X1;
Y_2 = EV(:,index)'*X2;


figure;
hold on
plot(Y_1, 0, 'r+');
plot(Y_2,0, 'b+');
xlabel('x')
ylabel('y')

figure;
hold on
hist(Y_1);
hist(Y_2);
h = findobj(gca,'Type','patch');
set(h(2), 'FaceColor','r','EdgeColor','w')
set(h(1), 'FaceColor','b','EdgeColor','w')

grid
xlabel('x')
ylabel('#')

