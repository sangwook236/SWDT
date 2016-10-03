method = 'SAD';
[dispMap, timeTaken]=denseMatch('tsukubaleft.jpg', 'tsukubaright.jpg', 9, 0, 16, method);

[rows, cols] = size(dispMap);
[X, Y] = meshgrid(1:cols, 1:rows);
surf(X, Y, dispMap, 'LineStyle', 'none');
axis equal;
view([0, 0, -1]);
