%----------------------------------------------------------
% MAP estimates for HMM w/ Gaussian Observations
% [ref] ${PMTK_HOME}/demos/hmmGaussTest.m

%----------------------------------------------------------

% [Caution]
%	${PMTK_HOME}/initPmtk3.m 실행 후, run을 사용하여 M file을 실행하면 정상적으로 실행되지 않음.
%	PMTK library가 run 함수를 제공하고 있어서, Matlab에서 제공하는 기본 run 함수가 정상적으로 동작하지 않음.
%		${PMTK_HOME}/projects/Emt/FA/BXPCA/run.m

% at desire.kaist.ac.kr
%run('D:\work_center\sw_dev\matlab\rnd\src\probabilistic_graphical_model\pmtk\pmtk3-1nov12\initPmtk3.m');
%cd('D:\work_center\sw_dev\matlab\rnd\test\probabilistic_graphical_model\pmtk');

%----------------------------------------------------------
loadData('speechDataDigits4And5');
data = [train4'; train5'];
d = 13;
nstates = 2;

% If type is 'gauss', or 'mixGaussTied', emissionPrior is a struct 
% with the parameters of a normal-inverse-Wishart distribution, namely, mu, Sigma, dof, k.
% test with a bogus prior
if 1
    emissionPrior.mu = ones(1, d);
    emissionPrior.Sigma = 0.1 * eye(d);
    emissionPrior.k = d;
    emissionPrior.dof = emissionPrior.k + 1;
else 
    emissionPrior.mu = [1 3 5 2 9 7 0 0 0 0 0 0 1];
    emissionPrior.Sigma = randpd(d) + eye(d);
    emissionPrior.k = 12;
    emissionPrior.dof = 15;
end

model = hmmFitEm(data, nstates, 'gauss', 'verbose', true, ...
	'piPrior', [3 2], 'emissionPrior', emissionPrior, ...
	'nRandomRestarts', 2, 'maxIter', 10);

X = hmmSample(model, 200, 10);
%model2 = hmmFitEm(X, 5, 'gauss', 'verbose', true);
