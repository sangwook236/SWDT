img = imread('horse5.bmp');
%figure; imshow(img);

b = bwboundaries(img);
b = b{4};

L = size(b, 1);

s = [ b(:,2), b(:,1) ];
figure; plot(s(:,1), s(:,2), 'k', 'LineWidth', 2);

fd = fourier_descriptor(s);
s2 = inverse_fourier_descriptor(fd, L / 2);

figure; plot(s2(:,1), s2(:,2), 'k', 'LineWidth', 2);
