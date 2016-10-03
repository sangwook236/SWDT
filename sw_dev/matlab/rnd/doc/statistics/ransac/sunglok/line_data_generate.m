function [data, inlier, outlier] = line_data_generate(model, numData, dataParam);
xBound = dataParam(1:2);
yBound = dataParam(3:4);
inlierRatio = dataParam(5);
inlierNoiseMean = dataParam(6);
inlierNoiseSTD = dataParam(7);
inlierNum = floor(numData * inlierRatio);
outlierNum = numData - inlierNum;

inlier = zeros(inlierNum,2);
for i = 1:inlierNum
    if model(2) == 0
        inlier(i,1) = -model(3) / model(1);
        inlier(i,2) = (yBound(2) - yBound(1))*rand() + yBound(1);
    else
        inlier(i,1) = (xBound(2) - xBound(1))*rand() + xBound(1);
        inlier(i,2) = -(model(1) * inlier(i,1) + model(3)) / model(2);
    end
    % Gaussian noise
    inlier(i,1) = inlier(i,1) + inlierNoiseSTD*randn() + inlierNoiseMean;
    inlier(i,2) = inlier(i,2) + inlierNoiseSTD*randn() + inlierNoiseMean;
end

outlier = zeros(outlierNum,2);
for i = 1:outlierNum
    % Random noise
    outlier(i,1) = (xBound(2) - xBound(1))*rand() + xBound(1);
    outlier(i,2) = (yBound(2) - yBound(1))*rand() + yBound(1);
end

data = [inlier;outlier];

