function [minModel, record] = msac(data, modelMake, modelNumSample, modelEvaluate, methodParam)
paramTry = methodParam{1};
paramError2 = methodParam{2}^2;

numData = size(data,1);
minPenalty = inf;
minModel = [];

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

    % 3. Evaluate the model
    penalty = 0;
    for j = 1:numData
        error2 = feval(modelEvaluate, model, data(j,:))^2;
        if (error2 < paramError2)
            penalty = penalty + error2;
        else
            penalty = penalty + paramError2;
        end
    end
    if (penalty < minPenalty)
        minPenalty = penalty;
        minModel = model;
    end
end

record.trials = i;
record.gamma = 0;
record.sigma = 0;

