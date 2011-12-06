% two-link arm dataset
% 25 binary images of size 128x96

Ndataset = 25;

IMG_WIDTH = 96;
IMG_HEIGHT = 128;

X = zeros(IMG_WIDTH * IMG_HEIGHT, Ndataset);
for ii = 1:25
	img_filename = sprintf('two_link_arm_%02d.png', ii);
	img = im2bw(imread(img_filename));
	X(:,ii) = img(:);
end;

save('two_link_arm_dataset.mat', 'X');
Y = load('two_link_arm_dataset.mat');
