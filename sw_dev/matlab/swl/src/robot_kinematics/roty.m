function rot = roty(rad)

c = cos(rad);
s = sin(rad);
rot = [
	c	0	s
	0	1	0
	-s	0	c
];
