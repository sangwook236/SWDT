% "The Global Positioning System and Inertial Navigation" by Jay Farrel & Matthew Barth
% Example p. 179

H = [
0.4513 -0.3543 -0.8190 1
-0.5018 0.5502 -0.6674 1
-0.6827 -0.6594 -0.3147 1
-0.3505 -0.4867 -0.8001 1
];

rho = [
3316.75
3268.47
3330.98
3346.88
];

delta_DGPS = [
12.52
60.86
0.93
-17.24
];

x1 = inv(H) * rho;
x2 = inv(H' * H) * H' * (rho + delta_DGPS);
