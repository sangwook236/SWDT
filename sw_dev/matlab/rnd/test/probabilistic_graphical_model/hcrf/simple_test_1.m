%----------------------------------------------------------
%addpath('D:\work\sw_dev\cpp\rnd\src\probabilistic_graphical_model\crf\hcrf\HCRF2.0b\bin\openMP');
%addpath('D:\work\sw_dev\cpp\rnd\src\probabilistic_graphical_model\crf\hcrf\HCRF2.0b\samples\matlab');

%----------------------------------------------------------
load sampleData;

matHCRF('createToolbox', 'crf', 'lbfgs', 0, 0);

intLabels = cellInt32(trainLabels);
matHCRF('setData', trainSeqs, intLabels);

matHCRF('train');

[modelCRF.model modelCRF.features] = matHCRF('getModel');
