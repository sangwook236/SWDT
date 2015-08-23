function [model] = line_lsm(data)
numData = size(data,1);
avgX = sum(data(:,1)) / numData;
avgY = sum(data(:,2)) / numData;
avgX2 = sum(data(:,1).^2) / numData;
avgY2 = sum(data(:,2).^2) / numData;
avgXY = sum(data(:,1).*data(:,2)) / numData;

A = [avgX2-avgX^2 avgXY-avgX*avgY; avgXY-avgX*avgY avgY2-avgY^2];
[vec,val] = eig(A);

a = vec(1,1);
b = vec(2,1);
c = -a * avgX - b*avgY;

model = [a,b,c];

