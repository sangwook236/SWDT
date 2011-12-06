function [ T ] = calc_robot_pose(dh_param, notation, joint_type, q)

numLink = size(dh_param, 1);

T = cell(1,numLink+1);
T{1} = eye(4);

% using Craig's notation
%	link transformation -> joint transformation
if strcmpi(notation, 'craig') == true
	for idx = 1:numLink
		a = dh_param(idx,1);
		alpha = dh_param(idx,2);
		if joint_type == 1  % revolute joint
			d = dh_param(idx,3);
			theta = dh_param(idx,4) + q(idx);
		else  % prismatic joint
			d = dh_param(idx,3) + q(idx);
			theta = dh_param(idx,4);
		end;
		
		LL = [ 1 0 0 a ; 0 cos(alpha) -sin(alpha) 0 ; 0 sin(alpha) cos(alpha) 0 ; 0 0 0 1 ];
		JJ = [ cos(theta) -sin(theta) 0 0 ; sin(theta) cos(theta) 0 0 ; 0 0 1 d ; 0 0 0 1 ];
		T{idx+1} = T{idx} * LL * JJ;
	end;
% using Paul's notation
%	joint transformation -> link transformation
elseif strcmpi(notation, 'paul') == true
	for idx = 1:numLink
		a = dh_param(idx,1);
		alpha = dh_param(idx,2);
		if joint_type == 1  % revolute joint
			d = dh_param(idx,3);
			theta = dh_param(idx,4) + q(idx);
		else  % prismatic joint
			d = dh_param(idx,3) + q(idx);
			theta = dh_param(idx,4);
		end;
		
		JJ = [ cos(theta) -sin(theta) 0 0 ; sin(theta) cos(theta) 0 0 ; 0 0 1 d ; 0 0 0 1 ];
		LL = [ 1 0 0 a ; 0 cos(alpha) -sin(alpha) 0 ; 0 sin(alpha) cos(alpha) 0 ; 0 0 0 1 ];
		T{idx+1} = T{idx} * JJ * LL;
	end;
else
	error('incorrect notation');
end
