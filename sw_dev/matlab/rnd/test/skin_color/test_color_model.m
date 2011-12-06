%--------------------------------------------------------------------
img = imread('hand_33.jpg');
%img = imread('hand_34.jpg');
%img = imread('hand_35.jpg');
%img = imread('hand_36.jpg');

theta = 0.5;  % threshold
mask_gmm = gmm_color_model(img, theta);

mode = 0;
mask_peer = peer_color_model(img, mode);

[rows, cols, channels] = size(img);
img_masked = img;
for ii = 1:rows
	for jj = 1:cols
		%if mask_gmm(ii,jj) == 0 || mask_peer(ii,jj) == 0
		if mask_gmm(ii,jj) == 0
		%if mask_peer(ii,jj) == 0
			img_masked(ii,jj,:) = [ 0 ; 0 ; 0 ];
		end;
	end;
end;

figure; imshow(img);
figure; imshow(img_masked);
