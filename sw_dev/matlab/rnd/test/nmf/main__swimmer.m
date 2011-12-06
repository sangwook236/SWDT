% swimmer dataset
% 256 binary images of size 32x32
% 256 images = 4 x 4 x 4 x 4

swimmer_data = load('swimmer_dataset.mat');

IMG_WIDTH = 32;
IMG_HEIGHT = 32;

dim = size(swimmer_data.Y);
swimmer_data.X = zeros(dim(1) * dim(2), dim(3));

for ii = 1:dim(3)
	swimmer_data.X(:,ii) = reshape(swimmer_data.Y(:,:,ii),[],1);
end;

imshow(reshape(swimmer_data.X(:,1) / max(swimmer_data.Y(:)), IMG_HEIGHT, IMG_WIDTH));

%--------------------------------------------------------------------
% using NMF: DTU Toolbox

%addpath('D:\work_center\sw_dev\matlab\rnd\src\nmf\dtu_toolbox\nmf_toolbox_ver1.4');

K = 49;
maxiter = 100;
%alg = 'mm';
alg = 'cjlin';
%alg = 'prob';
%alg = 'als';
%alg = 'alsobs';
[W, H] = nmf(swimmer_data.X, K, alg, maxiter);

PLOT_ROW = 7;
PLOT_COL = 7;
figure;
hold on;
for ii = 1:K
	subplot(PLOT_ROW, PLOT_COL, ii);
	imshow(reshape(W(:,ii), IMG_HEIGHT, IMG_WIDTH));
end;
hold off;

%--------------------------------------------------------------------
% using Projective NMF (PNMF)

K = 36;
maxiter = 100;
tol = 1.0e-5;
threshold = 0;
[W, H] = pnmf(swimmer_data.X, K, maxiter, tol, threshold);

maxW = max(W(:));

PLOT_ROW = 6;
PLOT_COL = 6;
figure;
hold on;
for ii = 1:K
	subplot(PLOT_ROW, PLOT_COL, ii);
	imshow(reshape(W(:,ii) / maxW, IMG_HEIGHT, IMG_WIDTH));
end;
hold off;

figure;
maxWtW = max(max(W' * W));
imshow(W' * W);

%--------------------------------------------------------------------
% using ARDPNMF: ARD + Projective NMF

K = 36;
maxiter = 100;
tol = 1.0e-5;
threshold = 0;
[W, H] = ardpnmf(swimmer_data.X, K, maxiter, tol, threshold);

maxW = max(W(:));

PLOT_ROW = 6;
PLOT_COL = 6;
figure;
hold on;
for ii = 1:K
	subplot(PLOT_ROW, PLOT_COL, ii);
	imshow(reshape(W(:,ii) / maxW, IMG_HEIGHT, IMG_WIDTH));
end;
hold off;

figure;
maxWtW = max(max(W' * W));
imshow(W' * W);

%--------------------------------------------------------------------
% using Orthogonal NMF by Seungjin Choi

K = 36;
maxiter = 100;
tol = 1.0e-5;
threshold = 0;
[W, H] = onmf_a(swimmer_data.X, K, maxiter, tol, threshold);
%[W, H] = onmf_s(swimmer_data.X, K, maxiter, tol, threshold);

maxW = max(W(:));

PLOT_ROW = 6;
PLOT_COL = 6;
figure;
hold on;
for ii = 1:K
	subplot(PLOT_ROW, PLOT_COL, ii);
	imshow(reshape(W(:,ii) / maxW, IMG_HEIGHT, IMG_WIDTH));
end;
hold off;

figure;
maxWtW = max(max(W' * W));
imshow(W' * W / maxWtW);
