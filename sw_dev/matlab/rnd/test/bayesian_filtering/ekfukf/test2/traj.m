function [ p v a ] = traj(t, coeffs)

% p(t) = a1 * t + a2 * sin(t) + a3 * sin(2*t) + a4 * sin(3*t) + a5 * cos(t) + a6 * cos(2*t) + a7 * cos(3*t)
p = coeffs(1) * t + coeffs(2) * sin(t) + coeffs(3) * sin(2*t) + coeffs(4) * sin(3*t) + coeffs(5) * cos(t) + coeffs(6) * cos(2*t) + coeffs(7) * cos(3*t);
v = coeffs(1) + coeffs(2) * cos(t) + 2 * coeffs(3) * cos(2*t) + 3 * coeffs(4) * cos(3*t) - coeffs(5) * sin(t) - 2 * coeffs(6) * sin(2*t) - 3 * coeffs(7) * sin(3*t);
a = -coeffs(2) * sin(t) - 4 * coeffs(3) * sin(2*t) - 9 * coeffs(4) * sin(3*t) - coeffs(5) * cos(t) - 4 * coeffs(6) * cos(2*t) - 9 * coeffs(7) * cos(3*t);
