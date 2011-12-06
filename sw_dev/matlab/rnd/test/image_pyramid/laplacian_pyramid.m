img = imread('lena_rgb.bmp');
img_0 = rgb2gray(img);

%--------------------------------------------------------------------
% Gaussian pyramid

mode = 0;
if 0 == mode
	img_1 = impyramid(img_0, 'reduce');
	img_2 = impyramid(img_1, 'reduce');
	img_3 = impyramid(img_2, 'reduce');
	img_4 = impyramid(img_3, 'reduce');

	img_1_exp = imresize(impyramid(img_1, 'expand'), size(img_0));
	img_2_exp = imresize(impyramid(img_2, 'expand'), size(img_1));
	img_3_exp = imresize(impyramid(img_3, 'expand'), size(img_2));
	img_4_exp = imresize(impyramid(img_4, 'expand'), size(img_3));
else
	img_1 = impyramid(img_0, 'reduce');
	img_1_exp = imresize(impyramid(img_1, 'expand'), size(img_0));

	img_2 = impyramid(img_1_exp, 'reduce');
	img_2_exp = imresize(impyramid(img_2, 'expand'), size(img_1_exp));

	img_3 = impyramid(img_2_exp, 'reduce');
	img_3_exp = imresize(impyramid(img_3, 'expand'), size(img_2_exp));

	img_4 = impyramid(img_3_exp, 'reduce');
	img_4_exp = imresize(impyramid(img_4, 'expand'), size(img_3_exp));
end;

size(img_1)
size(img_2)
size(img_3)
size(img_4)
size(img_1_exp)
size(img_2_exp)
size(img_3_exp)
size(img_4_exp)

figure;
hold on;
subplot(2,2,1);
imshow(img_0);
subplot(2,2,2);
imshow(img_1_exp);
subplot(2,2,3);
imshow(img_2_exp);
subplot(2,2,4);
imshow(img_3_exp);
hold off;

%--------------------------------------------------------------------
% Laplacing pyramid

if 0 == mode
	lap_1 = img_0 - img_1_exp;
	lap_2 = img_1 - img_2_exp;
	lap_3 = img_2 - img_3_exp;
	lap_4 = img_3 - img_4_exp;
else
	lap_1 = img_0 - img_1_exp;
	lap_2 = img_1_exp - img_2_exp;
	lap_3 = img_2_exp - img_3_exp;
	lap_4 = img_3_exp - img_4_exp;
end;

lap_1_he = adapthisteq(lap_1);
lap_2_he = adapthisteq(lap_2);
lap_3_he = adapthisteq(lap_3);
lap_4_he = adapthisteq(lap_4);

figure;
hold on;
subplot(2,2,1);
imshow(lap_1_he);
subplot(2,2,2);
imshow(lap_2_he);
subplot(2,2,3);
imshow(lap_3_he);
subplot(2,2,4);
imshow(lap_4_he);
hold off;
