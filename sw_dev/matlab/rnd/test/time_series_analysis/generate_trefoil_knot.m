function [x, y, z] = generate_trefoil_knot(t)

% trefoil knot
x = (2 + cos(3 .* t)) .* cos(2 .* t);
y = (2 + cos(3 .* t)) .* sin(2 .* t);
z = sin(3 .* t);

%plot3(x, y, z, '-');
