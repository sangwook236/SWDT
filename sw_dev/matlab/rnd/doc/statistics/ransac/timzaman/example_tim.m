%Tim Zaman (1316249), 2010, TU Delft, MSc:ME:BME:BMD:BR
clc
close all
clear all
 
%Define variables
nrs=100; %amount of numbers
maxnr=20; %max amount of X or Y
 
n(1,:)=[1 1 1];  %1st plane normal vector
n(2,:)=[1 1 -1]; %2nd plane normal vector
n(3,:)=[-1 0 1]; %3rd plane normal vector
n(4,:)=[1 -1 1]; %4th plane normal vector
ro=[10 20 10 40];
 
%Make random points...
for i=1:4 %for 4 different planes

	%Declarate 100 random x's and y's (p1 and p2)
	p(1:nrs,1:2,i)=randi(maxnr,nrs,2);
	%From the previous, calculate the Z
	p(1:nrs,3,i)=(ro(i)-n(i,1)*p(1:nrs,1,i)-n(i,2).*p(1:nrs,2,i))/n(i,3);

	%Add some random points
	for ii=1:20 %10 points
		randpt=p(randi(nrs),1:3,i);  %take an random existing point
		randvar=(randi(14,1,3)-7);       %adapt that randomly +-7
		p(nrs+ii,1:3,i)=randpt+randvar; %put behind pointmatrix
	end

end
 
%combine the four dataset-planes we have made into one
p_tot=[p(:,:,1);p(:,:,2);p(:,:,3);p(:,:,4)];
 
figure
plot3(p(:,1,1),p(:,2,1),p(:,3,1),'.r')
hold on; grid on
plot3(p(:,1,2),p(:,2,2),p(:,3,2),'.g')
plot3(p(:,1,3),p(:,2,3),p(:,3,3),'.b')
plot3(p(:,1,4),p(:,2,4),p(:,3,4),'.k')
 
no=3;%smallest number of points required
k=5;%number of iterations
t=2;%threshold used to id a point that fits well
d=70;%number of nearby points required
 
%Initialize samples to pick from
samples_pick=[1:1:length(p_tot)];
p_pick=p_tot;
for i=1:4 %Search for 4 planes
	[p_best,n_best,ro_best,X_best,Y_best,Z_best,error_best,samples_used]=local_ransac_tim(p_pick,no,k,t,d);

	samples_pick=[1:1:length(p_pick)];

	%Remove just used points from points for next plane
	for ii=1:length(samples_used) %In lack for a better way to do it used a loop
		samples_pick=samples_pick(samples_pick~=samples_used(ii)); %remove first
	end

	p_pick=p_pick(samples_pick,:);

	pause(.5)
	plot3(p_best(:,1),p_best(:,2),p_best(:,3),'ok')
	mesh(X_best,Y_best,Z_best);colormap([.8 .8 .8])
	beep
end
