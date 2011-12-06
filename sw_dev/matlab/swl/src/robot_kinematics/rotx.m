function rot = rotx(rad)

c = cos(rad);
s = sin(rad);
rot = [
    1	0	0
	0	c	-s
	0	s	c
];
