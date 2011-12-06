function rot = rot(axis, rad)

len = norm(axis);

eps = 1e-20;
if len <= eps
	R = eye(3);
	return;
end;

c = cos(rad);
s = sin(rad);
v = 1 - c;

kx = axis(1) / len;
ky = axis(2) / len;
kz = axis(3) / len;

rot = [
    kx * kx * v + c       kx * ky * v - kz * s  kx * kz * v + ky * s
	kx * ky * v + kz * s  ky * ky * v + c       ky * kz * v - kx * s
	kx * kz * v - ky * s  ky * kz * v + kx * s  kz * kz * v + c
];
