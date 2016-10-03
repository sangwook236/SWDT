% S-CURVE DATASET

  N=2000;
  K=12;
  d=2; 

clf; colordef none; colormap jet; set(gcf,'Position',[200,400,620,200]);

% PLOT TRUE MANIFOLD
  tt = [-1:0.1:0.5]*pi; uu = tt(end:-1:1); hh = [0:0.1:1]*5;
  xx = [cos(tt) -cos(uu)]'*ones(size(hh));
  yy = ones(size([tt uu]))'*hh;
  zz = [sin(tt) 2-sin(uu)]'*ones(size(hh));
  cc = [tt uu]' * ones(size(hh));

  subplot(1,3,1); cla;
  surf(xx,yy,zz,cc);
  view([12 -20 3]); grid off; axis off; hold on;
  lnx=-1*[1,1,1;1,-1,1]; lny=[0,0,0;5,0,0]; lnz=-1*[1,1,1;1,1,-1];
  lnh=line(lnx,lny,lnz);
  set(lnh,'Color',[1,1,1],'LineWidth',2,'LineStyle','-','Clipping','off');
  axis([-1,1,0,5,-1,3]); drawnow;

% GENERATE SAMPLED DATA
  angle = pi*(1.5*rand(1,N/2)-1); height = 5*rand(1,N);
  X = [[cos(angle), -cos(angle)]; height;[ sin(angle), 2-sin(angle)]];

% SCATTERPLOT OF SAMPLED DATA
  subplot(1,3,2); cla;
  scatter3(X(1,:),X(2,:),X(3,:),12,[angle angle],'+','filled');
  view([12 -20 3]); grid off; axis off;  hold on;
  lnh=line(lnx,lny,lnz);
  set(lnh,'Color',[1,1,1],'LineWidth',2,'LineStyle','-','Clipping','off');
  axis([-1,1,0,5,-1,3]); drawnow;

% RUN LLE ALGORITHM
Y=lle(X,K,d);

% SCATTERPLOT OF EMBEDDING
  subplot(1,3,3); cla;
  scatter(Y(1,:),Y(2,:),12,[angle angle],'+','filled');
  grid off; 
  set(gca,'XTick',[]); set(gca,'YTick',[]); 

