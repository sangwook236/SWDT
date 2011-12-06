function rot = rotz(rad)

c = cos(rad);
s = sin(rad);
rot = [
	c	-s	0
	s	c	0
	0	0	1
];
