function [x, y, z] = generate_simple_data_for_hmm(t)

% [ref] "Generalized Linear Gaussian Models" by A-V.I. Rosti and M.J.F. Gales
% pp. 4

% PMTK: latent variable models
hmmCreate
hmmFit
hmmInferNodes
hmmLogprob
hmmMap
hmmSample
hmmSamplePost
hmmToDgm
