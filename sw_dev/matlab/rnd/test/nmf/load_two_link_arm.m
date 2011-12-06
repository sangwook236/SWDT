% two-link arm dataset
% 25 binary images of size 128x96

two_link_arm_dataset = load('two_link_arm_dataset.mat');
dataset = two_link_arm_dataset.X;

Ndataset = size(dataset, 2);

IMG_WIDTH = 96;
IMG_HEIGHT = 128;

figure;
PLOT_ROW = ceil(sqrt(Ndataset));
PLOT_COL = PLOT_ROW;
plot_mode = 0;
if 1 == plot_mode
	for ii = 1:Ndataset
		img = reshape(dataset(:,ii), IMG_HEIGHT, IMG_WIDTH);
		subplot(PLOT_ROW, PLOT_COL, ii);
		imshow(img, [0 1]);
	end;
else
	all_imgs = zeros(IMG_HEIGHT * PLOT_ROW, IMG_WIDTH * PLOT_COL);
	idx = 1;
	for ii = 1:PLOT_ROW
		for jj = 1:PLOT_COL
			if idx <= Ndataset
				all_imgs(((ii-1)*IMG_HEIGHT+1):(ii*IMG_HEIGHT),((jj-1)*IMG_WIDTH+1):(jj*IMG_WIDTH)) = reshape(dataset(:,idx), IMG_HEIGHT, IMG_WIDTH);
			end;
			idx = idx + 1;
		end;
	end;
	imshow(all_imgs, [0 1]);
end;
