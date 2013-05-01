%----------------------------------------------------------
% MAP estimates for HMM w/ tied mixture-of-von-Mises (MovM) observations

%----------------------------------------------------------
%run('D:\work_center\sw_dev\matlab\rnd\src\probabilistic_graphical_model\pmtk\pmtk3-1nov12\genpathPMTK.m');

loadData('speechDataDigits4And5');
data = [train4'; train5'];
d = 13;
nstates = 2;
nmix = 3;  % must specify nmix

% If type is 'gauss', or 'mixGaussTied', emissionPrior is a struct 
% with the parameters of a normal-inverse-Wishart distribution, namely, mu, Sigma, dof, k.
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

model = hmmFit(data, nstates, 'mixGaussTied', 'verbose', true, ...
	'piPrior', [3 2], 'emissionPrior', emissionPrior, ...
    'nRandomRestarts', 2, 'maxiter', 5, 'nmix', nmix); 

if 0
    localev = hmmObs2LocalEv(data);
    localev(1:3:end) = nan;  % use dgmTrain for arbitrarily missing local ev
    Tmax = 131;  % maximum length of the sequences.
    dgm = hmmToDgm(model, Tmax);

    dgm2 = dgmTrain(dgm, 'localev', localev, 'verbose', true);
end

% This doesn't actually test anything!
