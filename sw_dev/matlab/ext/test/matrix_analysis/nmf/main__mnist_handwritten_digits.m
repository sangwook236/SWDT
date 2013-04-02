mnist_dataset = load('mnist_all.mat');

DIGIT_ID = 2;  % 0 ~ 9

if 0 == DIGIT_ID
	train_dataset = mnist_dataset.train0;
	test_dataset = mnist_dataset.test0;
elseif 1 == DIGIT_ID
	train_dataset = mnist_dataset.train1;
	test_dataset = mnist_dataset.test1;
elseif 2 == DIGIT_ID
	train_dataset = mnist_dataset.train2;
	test_dataset = mnist_dataset.test2;
elseif 3 == DIGIT_ID
	train_dataset = mnist_dataset.train3;
	test_dataset = mnist_dataset.test3;
elseif 4 == DIGIT_ID
	train_dataset = mnist_dataset.train4;
	test_dataset = mnist_dataset.test4;
elseif 5 == DIGIT_ID
	train_dataset = mnist_dataset.train5;
	test_dataset = mnist_dataset.test5;
elseif 6 == DIGIT_ID
	train_dataset = mnist_dataset.train6;
	test_dataset = mnist_dataset.test6;
elseif 7 == DIGIT_ID
	train_dataset = mnist_dataset.train7;
	test_dataset = mnist_dataset.test7;
elseif 8 == DIGIT_ID
	train_dataset = mnist_dataset.train8;
	test_dataset = mnist_dataset.test8;
elseif 9 == DIGIT_ID
	train_dataset = mnist_dataset.train9;
	test_dataset = mnist_dataset.test9;
end;

Ntrain = size(train_dataset, 1);
Ntest = size(test_dataset, 1);

IMG_WIDTH = 28;
IMG_HEIGHT = 28;

%DIGIT_INDEX = 1;
%img = reshape(train_dataset(DIGIT_INDEX,:), IMG_HEIGHT, IMG_WIDTH);
%imshow(img, [0 255]);

X = double(train_dataset' / 255);
X = X(:,1:100);
[DATA_DIM, DATA_NUM] = size(X);

%--------------------------------------------------------------------
% using NMF: DTU Toolbox

%addpath('D:\work_center\sw_dev\matlab\rnd\src\nmf\dtu_toolbox\nmf_toolbox_ver1.4');

K = 80;
maxiter = 1000;
alg = 'mm';
%alg = 'cjlin';
%alg = 'prob';
%alg = 'als';
%alg = 'alsobs';
[W, H] = nmf(X, K, alg, maxiter);

figure;
hold on;
PLOT_ROW = 8;
PLOT_COL = 10;
for ii = 1:K
	subplot(PLOT_ROW, PLOT_COL, ii);
	imshow(reshape(W(:,ii), IMG_HEIGHT, IMG_WIDTH));
end;
hold off;

%--------------------------------------------------------------------
% learning mixture models by multiplicative updates

BASIS_NUM = 4;  % 1, 2, 4, 8, 16, 32, 64
LABEL_NUM = 1;

Y = zeros(DATA_NUM, LABEL_NUM);
Y(:,:) = 1;

maxiter = 1000;
tol = 1.0e-5;
[W, THETA] = mixture_model_by_multiplicative_updates(X, Y, BASIS_NUM, maxiter, tol);
