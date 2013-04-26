%----------------------------------------------------------
%addpath('D:\work_center\sw_dev\cpp\rnd\src\probabilistic_graphical_model\crf\hcrf\HCRF2.0b\bin\openMP');
%addpath('D:\work_center\sw_dev\cpp\rnd\src\probabilistic_graphical_model\crf\hcrf\HCRF2.0b\samples\matlab');

%----------------------------------------------------------
load sampleData;

paramsData.weightsPerSequence = ones(1,128) ;
paramsData.factorSeqWeights = 1;

%----------------------------------------------------------
disp('processing CRF ...');

paramsNodCRF.normalizeWeights = 1;
R{1}.params = paramsNodCRF;
tic;
[R{1}.model R{1}.stats] = train(trainSeqs, trainLabels, R{1}.params);
fprintf(1, 'elapsed time: %f (training)', toc)
tic;
[R{1}.ll R{1}.labels] = test(R{1}.model, testSeqs, testLabels);
fprintf(1, ', %f (testing)\n', toc)

%--S [] 2013/01/21: Sang-Wook Lee
matLabels = cell2mat(R{1}.labels);
matLikelihoods = cell2mat(R{1}.ll);
[R{1}.d R{1}.f] = CreateROC(matLabels, matLikelihoods(2,:), R{1}.params.rocRange);
%--E [] 2013/01/21: Sang-Wook Lee

%----------------------------------------------------------
disp('processing HCRF ...');

paramsNodHCRF.normalizeWeights = 1;
R{2}.params = paramsNodHCRF;
tic;
[R{2}.model R{2}.stats] = train(trainCompleteSeqs, trainCompleteLabels, R{2}.params);
fprintf(1, 'elapsed time: %f (training)', toc)
tic;
[R{2}.ll R{2}.labels] = test(R{2}.model, testSeqs, testLabels);
fprintf(1, ', %f (testing)\n', toc)

%--S [] 2013/01/21: Sang-Wook Lee
matLabels = cell2mat(R{2}.labels);
matLikelihoods = cell2mat(R{2}.ll);
[R{2}.d R{2}.f] = CreateROC(matLabels, matLikelihoods(2,:), R{2}.params.rocRange);
%--E [] 2013/01/21: Sang-Wook Lee

%----------------------------------------------------------
disp('processing LDCRF ...');

paramsNodLDCRF.normalizeWeights = 1;
R{3}.params = paramsNodLDCRF;
tic;
[R{3}.model R{3}.stats] = train(trainSeqs, trainLabels, R{3}.params);
fprintf(1, 'elapsed time: %f (training)', toc)
tic;
[R{3}.ll R{3}.labels] = test(R{3}.model, testSeqs, testLabels);
fprintf(1, ', %f (testing)\n', toc)

%--S [] 2013/01/21: Sang-Wook Lee
matLabels = cell2mat(R{3}.labels);
matLikelihoods = cell2mat(R{3}.ll);
[R{3}.d R{3}.f] = CreateROC(matLabels, matLikelihoods(2,:), R{3}.params.rocRange);
%--E [] 2013/01/21: Sang-Wook Lee

%----------------------------------------------------------
disp('plotting ROC ...');

plotResults(R);
