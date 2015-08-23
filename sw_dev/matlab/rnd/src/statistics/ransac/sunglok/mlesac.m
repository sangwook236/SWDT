function [minModel, record] = mlesac(data, modelMake, modelNumSample, modelEvaluate, methodParam)
paramTry = methodParam{1};
paramSigma2 = methodParam{2}^2;
paramEMTry = methodParam{3};
paramNu = methodParam{4};

[numData,dimData] = size(data);
error2 = zeros(numData,1);
minLogLikelihood = inf;
minModel = [];
minGamma = 0.5;

for i = 1:paramTry
    % 1. Sample data
    sampleIndex = logical(zeros(numData,1));
    while sum(sampleIndex) < modelNumSample
        index = floor(numData*rand()) + 1;
        sampleIndex(index) = true;
    end
    sampleData = data(sampleIndex,:);

    % 2. Estimate a model
    model = feval(modelMake, sampleData);

    % 3 + 4. Common stuff
    for j = 1:numData
        error = feval(modelEvaluate, model, data(j,:));
        error2(j) = error^2;
    end

    % 3. Estimate 'gamma' using EM
    gamma = 0.5;
    for j = 1:paramEMTry
        sumInlierProb = 0;
        probOutlier = (1 - gamma) / paramNu;
        probInlier_pre = gamma / sqrt(2 * pi * paramSigma2)^dimData;
        for k = 1:numData
            probInlier = probInlier_pre * exp(-0.5 * error2(k) / paramSigma2);
            probZ = probInlier / (probInlier + probOutlier);
            sumInlierProb = sumInlierProb + probZ;
        end
        gamma = sumInlierProb / numData;
    end

    % 4. Evaluate the model
    sumLogLikelihood = 0;
    probOutlier = (1 - gamma) / paramNu;
    probInlier_pre = gamma / sqrt(2 * pi * paramSigma2)^dimData;
    for j = 1:numData
        probInlier = probInlier_pre * exp(-0.5 * error2(j) / paramSigma2);
        likelihood = probInlier + probOutlier;
        sumLogLikelihood = sumLogLikelihood - log(likelihood);
    end
    if (sumLogLikelihood < minLogLikelihood)
        minLogLikelihood = sumLogLikelihood;
        minModel = model;
        minGamma = gamma;
    end
end

record.trials = i;
record.gamma = minGamma;
record.sigma = 0;

