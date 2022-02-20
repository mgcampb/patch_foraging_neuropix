%% Demo mvt

t = 0:.01:1000;
travel_time = 5;
rate_A = 2;
max_A = 3;
rate_B = 1;
max_B = 2;

% A = max_A ./ (1 + exp(-1 .* (t - travel_time)));
% B = max_B ./ (1 + exp(-1 .* (t - travel_time)));

A = max_A * sqrt(rate_A * 20 * (t - travel_time));
B = max_B * sqrt(rate_B * 20 * (t - travel_time));

figure();hold on
plot(t,A)
plot(t,B)