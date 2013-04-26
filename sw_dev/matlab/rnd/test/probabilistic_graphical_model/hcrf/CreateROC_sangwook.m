function [detect falsePos threshValue rawResults] = CreateROC(labels, ll, rangeThresh)

if iscell(ll) && numel(ll)>0
    if iscell(labels) && numel(labels)
        if numel(labels{1}) ~= size(ll{1},2)
            ll = transposeCellArray(ll)';
        end
    end
    ll = cell2mat(ll);
end
if iscell(labels) && numel(labels)
    labels = cell2mat(labels);
end

detect = zeros(1,numel(rangeThresh));
falsePos = zeros(1,numel(rangeThresh));
threshValue = zeros(1,numel(rangeThresh));
if numel(rangeThresh) == 1
    minValue = min(ll);
    maxValue = max(ll);
    inc = (maxValue - minValue)/abs(rangeThresh);
    rangeThresh = minValue:inc:maxValue;
end

%--S [] 2013/01/25: Sang-Wook Lee
target_class_label = 1;
%--E [] 2013/01/25: Sang-Wook Lee

rawResults = zeros(numel(rangeThresh),4);
for i=1:size(rangeThresh,2)
    thresh = rangeThresh(i);
    d = (ll > thresh);
    n = sum(d == 1 & labels == target_class_label);  % true positive (TP)
    f = sum(d == 1 & labels ~= target_class_label);  % false positive (FP)
    t = sum(labels == target_class_label);  % positive (P)
    totalfalsepos = sum(labels ~= target_class_label);  % negative (N)
    if t == 0
        detect(i) = 0;
    else
        detect(i) = n/t;
    end
    falsePos(i) = f/totalfalsepos;
    threshValue(i) = thresh;
    rawResults(i,:) = [n t f totalfalsepos];
end
