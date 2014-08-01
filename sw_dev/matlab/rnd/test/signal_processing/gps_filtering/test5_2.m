% "The Global Positioning System and Inertial Navigation" by Jay Farrel & Matthew Barth
% Example p. 163

H = [
-0.557466 0.829830 -0.024781 1
0.469913 0.860463 0.196942 1
0.086117 0.936539 -0.339823 1
0.661510 -0.318625 -0.678884 1
-0.337536 0.461389 -0.820482 1
0.762094 0.267539 -0.589606 1
];

V = inv(H' * H);
GDOP = sqrt(trace(V));
