% two-link arm dataset
% 25 binary images of size 128x96

two_link_arm_dataset = load('two_link_arm_dataset.mat');
dataset = two_link_arm_dataset.X;

Ndataset = size(dataset, 2);

IMG_WIDTH = 96;
IMG_HEIGHT = 128;

%--------------------------------------------------------------------
% using Projective NMF (PNMF)

K = 10;
maxiter = 100;
tol = 1.0e-5;
threshold = 0;
[W, H] = pnmf(dataset, K, maxiter, tol, threshold);

maxW = max(W(:));

figure;
hold on;
for ii = 1:K
	subplot(4, 4, ii);
	imshow(reshape(W(:,ii) / maxW, IMG_HEIGHT, IMG_WIDTH));
end;
hold off;

figure;
maxWtW = max(max(W' * W));
imshow(W' * W);
