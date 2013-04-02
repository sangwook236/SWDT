%----------------------------------------------------------
% create HMM model

A = [
	0.7 0.3
	0.4 0.6
];
C = [
	0.1 0.4 0.5
	0.6 0.3 0.1
];
emission = tabularCpdCreate(C);
init_pi = [0.6 0.4]';
model = hmmCreate('discrete', init_pi, A, emission);

%----------------------------------------------------------
% find the most-probable (Viterbi) path through the HMM state trellis

observed = [1 2 3];
viterbiPath = hmmMap(model, observed);
