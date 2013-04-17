%----------------------------------------------------------
% normal distribution (1-dim)

% target distribution
pdf = @(x) normpdf(x);

% proposal distribution, q(x | y)
delta = .5;
proppdf = @(x, y) unifpdf(y - x, -delta, delta);
proprnd = @(x) x + rand * 2 * delta - delta;

nsamples = 15000;
burn_in_period = 1000;
smpl1 = mhsample(1, nsamples, 'pdf', pdf, 'proprnd', proprnd, 'symmetric', 1, 'burnin', burn_in_period);

figure;
histfit(smpl1, 100);
h = get(gca, 'Children');
set(h(2), 'FaceColor', [.8 .8 1]);

%----------------------------------------------------------
% multivariate normal distribution (2-dim)

% [ref] https://code.google.com/p/pmtk/source/browse/trunk/pmtk/examples/mcmcExamples/mhMvn2d.m?r=296

%run('D:\work_center\sw_dev\matlab\rnd\src\probabilistic_graphical_model\pmtk\pmtk3-1nov12\genpathPMTK.m');
%addpath('D:\working_copy\swl_https\matlab\src\statistical_analysis');

% target distribution
mu = [1 1];
sigma = [1 0 ; 0 1];
pdf = @(x) mvnpdf(x, mu, sigma);

% proposal distribution, q(x | y) -> q(x)
delta = 0.5;
proppdf = @(x, y) @(x) mvnrnd(x, eye(2));
proprnd = @(x) x + (rand(1,2) * 2 * delta - delta);

nsamples = 100000;
burn_in_period = 1000;
smpl2 = mhsample([1 1], nsamples, 'pdf', pdf, 'proprnd', proprnd, 'symmetric', 1, 'burnin', burn_in_period);

xedges = linspace(-1, 3, 50);
yedges = linspace(-1, 3, 50); 
histmat = hist2(smpl2(:,1), smpl2(:,2), xedges, yedges);

figure;
pcolor(xedges, yedges, histmat');
colorbar;
axis square tight;

%----------------------------------------------------------
% gamma distribution (1-dim)

% target distribution
alpha = 2.43;
beta = 1;
pdf = @(x) gampdf(x, alpha, beta);

% proposal distribution, q(x | y) -> q(x)
proppdf = @(x, y) gampdf(x, floor(alpha), floor(alpha) / alpha);
proprnd = @(x) sum(exprnd(floor(alpha) / alpha, floor(alpha), 1));

nsamples = 10000;
burn_in_period = 1000;
smpl3 = mhsample(1, nsamples, 'pdf', pdf, 'proprnd', proprnd, 'proppdf', proppdf, 'burnin', burn_in_period);

figure;
%xxhat = cumsum(smpl3.^2) ./ (1:nsamples)';
%plot(1:nsamples, xxhat);
subplot(2,1,1), hist(smpl3, 100);
axis_val = axis;
subplot(2,1,2), ezplot(pdf, [axis_val(1) axis_val(2)]);

%----------------------------------------------------------
% von Mises distribution

% [ref] "A Bayesian Analysis of Directional Data Using the von Mises-Fisher Distribution", G. Nunez-Antonio and E. Gutierrez-Pena, CSSC, 2005.
% [ref] "Finding the Location of a Signal: A Bayesian Analysis", P. Guttorp and R. A. Lockhart, JASA, 1988.

%addpath('D:\working_copy\swl_https\matlab\src\directional_statistics');

% target distribution
R_n = 20;
theta_n = pi;
c = 5;
n = 100;
kappa0 = 1;
pdf = @(theta) exp(kappa0 * R_n * cos(theta - theta_n)) / besseli(0, kappa0)^(c + n);

% proposal distribution, q(x | y)
delta = .5;
%a_c = 1;
%b_c = 1;
%proppdf = @(theta, kappa) unifpdf(theta, 0, 2*pi) * gampdf(kappa0, a_c, b_c);
proppdf = @(x, y) unifpdf(y - x, -delta, delta);
%proprnd = @(x) x + rand * 2 * pi;
proprnd = @(x) x + rand * 2 * delta - delta;

nsamples = 15000;
burn_in_period = 1000;
smpl4 = mhsample(0, nsamples, 'pdf', pdf, 'proprnd', proprnd, 'proppdf', proppdf, 'burnin', burn_in_period);
smpl4 = mod(smpl4, 2*pi);

figure;
subplot(2,1,1), hist(smpl4, 100)
axis_val = axis;
axis([0 2*pi axis_val(3) axis_val(4)]);
subplot(2,1,2), ezplot(pdf, [0 2*pi]);
%subplot(2,1,2), ezplot(pdf, [axis_val(1) axis_val(2)]);
