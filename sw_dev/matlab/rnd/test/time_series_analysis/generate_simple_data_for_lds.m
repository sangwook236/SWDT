function [x, y, z] = generate_simple_data_for_lds(t)

% [ref] "Generalized Linear Gaussian Models" by A-V.I. Rosti and M.J.F. Gales
% pp. 3

% PMTK: latent variable models
ldsCreate
ldsFit
ldsInfer
ldsSample
