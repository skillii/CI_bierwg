function [] = plotErrorFunction(weights)

x = -2:0.1:2;
y = -2:0.1:2;
w1 = meshgrid(x, y);
w2 = w1';

error = getError(w1, w2);

figure;
contour(x, y, error, 30);

hold all;
handle(1) = plot(weights(:,1), weights(:,2), 'ro-', 'LineWidth', 3);

handle(2) = plot(weights(end,1), weights(end,2), 'ks', 'MarkerSize', 10, 'LineWidth', 3);
handle(3) = plot(weights(1,1), weights(1,2), 'ko', 'MarkerSize', 10, 'LineWidth', 3);

legend(handle, 'Weights', 'EndVector', 'StartVector', 'Location', 'SouthEast');

function [error] = getError(w1, w2)


error = - 2 * exp(-20 * (0.5*(w1 - 1).^2 +  (w2 - 1).^2)) - 1 *  exp(-0.1 * (2 * (w1 + 1).^2 *2 +  (w2 ).^2/2));
