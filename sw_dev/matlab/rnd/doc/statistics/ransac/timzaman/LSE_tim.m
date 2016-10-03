function [n_est ro_est X Y Z]=LSE_tim(p)
%¨Ï Tim Zaman 2010, input: p (points)
% Works like [n_est ro_est X Y Z]=LSE(p)
% p should be a Mx3; [points x [X Y Z]]</code>
 
%Calculate mean of all points
pbar=mean(p);
for i=1:length(p)
A(:,:,i)=(p(i,:)-pbar)'*(p(i,:)-pbar);
end
 
%Sum up all entries in A
Asum=sum(A,3);
[V ~]=eig(Asum);
 
%Calculate new normal vector
n_est=V(:,1);
 
%Calculate new ro
ro_est=dot(n_est,pbar);
 
[X,Y]=meshgrid(min(p(:,1)):max(p(:,1)),min(p(:,2)):max(p(:,2)));
Z=(ro_est-n_est(1)*X-n_est(2).*Y)/n_est(3);
end
