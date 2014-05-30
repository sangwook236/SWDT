% using Craig's notation
%             th    d       a    alpha
L(1) = Link([ 0     0       0    0        0 ], 'modified');
L(2) = Link([ 0     0       0    pi/2     0 ], 'modified');
L(3) = Link([ 0     0.295   0   -pi/2     0 ], 'modified');
L(4) = Link([ 0     0       0    pi/2     0 ], 'modified');
L(5) = Link([ 0     0       0   -pi/2     0 ], 'modified');
L(6) = Link([ 0     0.225   0    0        0 ], 'modified');

L(6).qlim = [0 0];

robot5r = SerialLink(L, 'name', '5R Robot');
robot5r.plotopt = { 'workspace', [ -2 2 -2 2 -2 2 ] };

qz = [ 0 0 0 0 0 0 ];
%robot5r.plot(qz);
robot5r.teach(qz);
