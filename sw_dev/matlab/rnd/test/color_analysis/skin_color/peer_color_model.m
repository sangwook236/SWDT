function [ mask ] = peer_color_model(img, mode)

% [ref] "Human Skin Colour Clustering for Face Detection"
% Peter Peer, Jure Kovac, & Franc Solina, Eurocon 2003

[rows, cols, channels] = size(img);
mask = zeros(rows, cols);

if mode == 1
	% the skin colour under flashlight or (light) daylight lateral illumination
	% R > 220 && G > 210 && B > 170 &&
	% abs(R - G) > 15 &&
	% R > G && R > B

	for ii = 1:rows
		for jj = 1:cols
			R = img(ii, jj, 1);
			G = img(ii, jj, 2);
			B = img(ii, jj, 3);

			if R > 220 && G > 210 && B > 170 && ...
				abs(R - G) > 15 && R > G && R > B
				mask(ii,jj) = 1;
			end;
		end;
	end;
else
	% the skin colour at uniform daylight illumination
	% R > 95 && G > 40 && B > 20 &&
	% max(R, G, B) - min(R, G, B) > 15 &&
	% abs(R - G) > 15 &&
	% R > G && R > B

	for ii = 1:rows
		for jj = 1:cols
			R = img(ii, jj, 1);
			G = img(ii, jj, 2);
			B = img(ii, jj, 3);

			if R > 95 && G > 40 && B > 20 && ...
				(max(img(ii, jj, :)) - min(img(ii, jj, :))) > 15 && ...
				abs(R - G) > 15 && R > G && R > B
				mask(ii,jj) = 1;
			end;
		end;
	end;
end;
