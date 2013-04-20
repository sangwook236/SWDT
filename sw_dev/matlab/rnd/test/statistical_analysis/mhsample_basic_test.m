%----------------------------------------------------------
% normal distribution (1-dim)

% target distribution
pdf = @(x) normpdf(x);

% proposal distribution, q(x | y)
delta = .5;
proppdf = @(x, y) unifpdf(y - x, -delta, delta);
proprnd = @(x) x + rand * 2 * delta - delta;

num_samples = 10000;
burn_in_period = 1000;
thinning_period = 5;
smpl1 = mhsample(1, num_samples, 'pdf', pdf, 'proprnd', proprnd, 'symmetric', 1, 'thin', thinning_period, 'burnin', burn_in_period);

figure;
histfit(smpl1, 100);
h = get(gca, 'Children');
set(h(2), 'FaceColor', [0.8 0.8 1]);

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

num_samples = 100000;
burn_in_period = 1000;
thinning_period = 5;
smpl2 = mhsample([1 1], num_samples, 'pdf', pdf, 'proprnd', proprnd, 'symmetric', 1, 'thin', thinning_period, 'burnin', burn_in_period);

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

num_samples = 10000;
burn_in_period = 1000;
thinning_period = 5;
smpl3 = mhsample(1, num_samples, 'pdf', pdf, 'proprnd', proprnd, 'proppdf', proppdf, 'thin', thinning_period, 'burnin', burn_in_period);

figure;
if 0
	%xxhat = cumsum(smpl3.^2) ./ (1:num_samples)';
	%plot(1:num_samples, xxhat);
	subplot(2,1,1), hist(smpl3, 100);
	axis_rng = axis;
	subplot(2,1,2), ezplot(pdf, [axis_rng(1) axis_rng(2)]);
elseif 0
	[binheight, bincenter] = hist(smpl3, 100);
	binwidth = bincenter(2) - bincenter(1);

	hold on;
	h1 = bar(bincenter, binheight, 'hist');
	set(h1, 'facecolor', [0.8 0.8 1]);
	axis_rng = axis;
	area = quad(pdf, axis_rng(1), axis_rng(2));
	h2 = ezplot(@(x) (num_samples * binwidth / area) * pdf(x), [axis_rng(1) axis_rng(2)]);
	set(h2, 'color', [1 0 0], 'linewidth', 2);
	hold off;
else
	[binheight, bincenter] = hist(smpl3, 100);
	binwidth = bincenter(2) - bincenter(1);
	x_rng = [ floor(min(bincenter)) ceil(max(bincenter)) ];
	%area1 = quad(pdf, x_rng(1), x_rng(2));
	area2 = num_samples * binwidth;

	% expectation of a function, g = E_f[g]
	% E_f[g] = 1/N sum(i=1 to N, g(x_i)) where x_i ~ pdf(x)
	%E =  mean(g(smpl4));

	hold on;
	h1 = bar(bincenter, binheight / area2, 'hist');
	set(h1, 'facecolor', [0.8 0.8 1]);
	axis_rng = axis;
	h2 = ezplot(pdf, x_rng);
	set(h2, 'color', [1 0 0], 'linewidth', 2);
	hold off;
end;

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

num_samples = 10000;
burn_in_period = 1000;
thinning_period = 5;
smpl4 = mhsample(0, num_samples, 'pdf', pdf, 'proprnd', proprnd, 'proppdf', proppdf, 'thin', thinning_period, 'burnin', burn_in_period);
smpl4 = mod(smpl4, 2*pi);

figure;
if 0
	subplot(2,1,1), hist(smpl4, 100)
	axis_rng = axis;
	axis([0 2*pi axis_rng(3) axis_rng(4)]);
	subplot(2,1,2), ezplot(pdf, [0 2*pi]);
	%subplot(2,1,2), ezplot(pdf, [axis_rng(1) axis_rng(2)]);
elseif 0
	[binheight, bincenter] = hist(smpl4, 50);
	binwidth = bincenter(2) - bincenter(1);
	area = quad(pdf, 0, 2*pi);

	hold on;
	h1 = bar(bincenter, binheight, 'hist');
	set(h1, 'facecolor', [0.8 0.8 1]);
	axis_rng = axis;
	h2 = ezplot(@(x) (num_samples * binwidth / area) * pdf(x), [0 2*pi]);
	set(h2, 'color', [1 0 0], 'linewidth', 2);
	axis([0 2*pi axis_rng(3) axis_rng(4)]);
	hold off;
else
	[binheight, bincenter] = hist(smpl4, 50);
	binwidth = bincenter(2) - bincenter(1);
	x_rng = [ 0 2*pi ];
	area1 = quad(pdf, x_rng(1), x_rng(2));
	area2 = num_samples * binwidth;

	% expectation of a function, g = E_f[g]
	% E_f[g] = 1/N sum(i=1 to N, g(x_i)) where x_i ~ pdf(x)
	%E =  mean(g(smpl5));

	hold on;
	h1 = bar(bincenter, binheight / area2, 'hist');
	set(h1, 'facecolor', [0.8 0.8 1]);
	axis_rng = axis;
	h2 = ezplot(@(x) pdf(x) / area1, x_rng);
	set(h2, 'color', [1 0 0], 'linewidth', 2);
	axis([x_rng(1) x_rng(2) axis_rng(3) axis_rng(4)]);
	hold off;
end;
