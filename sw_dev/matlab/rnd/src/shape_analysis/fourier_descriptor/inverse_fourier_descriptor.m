function s = inverse_fourier_descriptor(z, nd)

% [ref] "Digital Image Processing using Matlab" p. 459

np = length(z);
if nargin == 1 | nd > np
	nd = np;
end;

% create an alternating sequence of 1s and -1s for use in centering the transform
x = 0:(np - 1);
m = ((-1).^x)';

% use only nd descriptors in the inverse.
% since the descriptors are centered, (np - nd) / 2 terms from each end of the sequence are set to 0.
d = round((np - nd) / 2);
z(1:d) = 0;
z((np - d + 1):np) = 0;

% compute the inverse and convert back to coordinates
zz = ifft(z);
s(:,1) = real(zz);
s(:,2) = imag(zz);

% multiply the input sequence by alternating 1s and -1s to center the transform
s(:,1) = m .* s(:,1);
s(:,2) = m .* s(:,2);
