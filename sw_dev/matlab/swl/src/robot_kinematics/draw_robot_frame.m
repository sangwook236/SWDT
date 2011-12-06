function draw_robot_frame(T, axis_length)

numFrame = length(T);

hold on;
axis equal;

for idx=1:numFrame
	plot3([T{idx}(1,4) ; T{idx}(1,4) + axis_length * T{idx}(1,1)], [T{idx}(2,4) ; T{idx}(2,4) + axis_length * T{idx}(2,1)], [T{idx}(3,4) ; T{idx}(3,4) + axis_length * T{idx}(3,1)], 'r-');
	plot3([T{idx}(1,4) ; T{idx}(1,4) + axis_length * T{idx}(1,2)], [T{idx}(2,4) ; T{idx}(2,4) + axis_length * T{idx}(2,2)], [T{idx}(3,4) ; T{idx}(3,4) + axis_length * T{idx}(3,2)], 'g-');
	plot3([T{idx}(1,4) ; T{idx}(1,4) + axis_length * T{idx}(1,3)], [T{idx}(2,4) ; T{idx}(2,4) + axis_length * T{idx}(2,3)], [T{idx}(3,4) ; T{idx}(3,4) + axis_length * T{idx}(3,3)], 'b-');
end;

xlabel('x');
ylabel('y');
zlabel('z');

hold off;
