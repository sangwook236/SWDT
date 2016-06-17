%
% SAMMON_TEST - minimal test harness for Sammon mapping routine
%
%    SAMMON_TEST provides a simple test harness and demonstration program
%    for SAMMON, a vectorised MATLAB implementation of Sammon's nonlinear
%    mapping algorithm [1].  Basically, it just generates and displays a
%    Sammon map of Fisher's famous Iris dataset [2].
%
%    References :
%
%       [1] Sammon, John W. Jr., "A Nonlinear Mapping for Data Structure
%           Analysis", IEEE Transactions on Computers, vol. C-18, no. 5,
%           pp 401-409, May 1969.
%
%       [2] Fisher, R. A., "The use of multiple measurements in taxonomic
%           problems", Annual Eugenics, vol. 7, part II, pp 179-188, 1936.
%
%    See also : SAMMON

%
% File        : sammon_test.m
%
% Date        : Monday 12th November 2007
%
% Author      : Gavin C. Cawley 
%
% Description : Minimal test harness for a vectorised MATLAB implementation of
%               Sammon's nonlinear mapping algorithm [1] (sammon.m).  
%               Basically, it just creates and displays a Sammon mapping of
%               Fisher's famous Iris dataset [2].
%
% References  : [1] Sammon, John W. Jr., "A Nonlinear Mapping for Data
%                   Structure Analysis", IEEE Transactions on Computers,
%                   vol. C-18, no. 5, pp 401-409, May 1969.
%
%               [2] Fisher, R. A., "The use of multiple measurements in
%                   taxonomic problems", Annual Eugenics, vol. 7, part II,
%                   pp 179-188, 1936.
%
% History     : 11/08/2004 - v1.00
%               12/12/2007 - v1.10 - PCA based initialisation
%
% Copyright   : (c) Dr Gavin C. Cawley, November 2007.
%
%    This program is free software; you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation; either version 2 of the License, or
%    (at your option) any later version.
%
%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with this program; if not, write to the Free Software
%    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
%

% start from a clean slate

clear all;

%
% this is Fisher's Iris dataset
%
% Attribute information:
%
%    1 - sepal length in cm
%    2 - sepal width in cm
%    3 - petal length in cm
%    4 - petal width in cm
%    5 - class :
%
%           Iris setosa      = 1
%           Iris Versicolour = 2
%           Iris Virginica   = 3
%

iris = [ 5.1 3.5 1.4 0.2 1 ; 4.9 3.0 1.4 0.2 1 ; 4.7 3.2 1.3 0.2 1 ;
         4.6 3.1 1.5 0.2 1 ; 5.0 3.6 1.4 0.2 1 ; 5.4 3.9 1.7 0.4 1 ;
         4.6 3.4 1.4 0.3 1 ; 5.0 3.4 1.5 0.2 1 ; 4.4 2.9 1.4 0.2 1 ;
         4.9 3.1 1.5 0.1 1 ; 5.4 3.7 1.5 0.2 1 ; 4.8 3.4 1.6 0.2 1 ;
         4.8 3.0 1.4 0.1 1 ; 4.3 3.0 1.1 0.1 1 ; 5.8 4.0 1.2 0.2 1 ;
         5.7 4.4 1.5 0.4 1 ; 5.4 3.9 1.3 0.4 1 ; 5.1 3.5 1.4 0.3 1 ;
         5.7 3.8 1.7 0.3 1 ; 5.1 3.8 1.5 0.3 1 ; 5.4 3.4 1.7 0.2 1 ;
         5.1 3.7 1.5 0.4 1 ; 4.6 3.6 1.0 0.2 1 ; 5.1 3.3 1.7 0.5 1 ;
         4.8 3.4 1.9 0.2 1 ; 5.0 3.0 1.6 0.2 1 ; 5.0 3.4 1.6 0.4 1 ;
         5.2 3.5 1.5 0.2 1 ; 5.2 3.4 1.4 0.2 1 ; 4.7 3.2 1.6 0.2 1 ;
         4.8 3.1 1.6 0.2 1 ; 5.4 3.4 1.5 0.4 1 ; 5.2 4.1 1.5 0.1 1 ;
         5.5 4.2 1.4 0.2 1 ; 4.9 3.1 1.5 0.1 1 ; 5.0 3.2 1.2 0.2 1 ;
         5.5 3.5 1.3 0.2 1 ; 4.9 3.1 1.5 0.1 1 ; 4.4 3.0 1.3 0.2 1 ;
         5.1 3.4 1.5 0.2 1 ; 5.0 3.5 1.3 0.3 1 ; 4.5 2.3 1.3 0.3 1 ;
         4.4 3.2 1.3 0.2 1 ; 5.0 3.5 1.6 0.6 1 ; 5.1 3.8 1.9 0.4 1 ;
         4.8 3.0 1.4 0.3 1 ; 5.1 3.8 1.6 0.2 1 ; 4.6 3.2 1.4 0.2 1 ;
         5.3 3.7 1.5 0.2 1 ; 5.0 3.3 1.4 0.2 1 ; 7.0 3.2 4.7 1.4 2 ;
         6.4 3.2 4.5 1.5 2 ; 6.9 3.1 4.9 1.5 2 ; 5.5 2.3 4.0 1.3 2 ;
         6.5 2.8 4.6 1.5 2 ; 5.7 2.8 4.5 1.3 2 ; 6.3 3.3 4.7 1.6 2 ;
         4.9 2.4 3.3 1.0 2 ; 6.6 2.9 4.6 1.3 2 ; 5.2 2.7 3.9 1.4 2 ;
         5.0 2.0 3.5 1.0 2 ; 5.9 3.0 4.2 1.5 2 ; 6.0 2.2 4.0 1.0 2 ;
         6.1 2.9 4.7 1.4 2 ; 5.6 2.9 3.6 1.3 2 ; 6.7 3.1 4.4 1.4 2 ;
         5.6 3.0 4.5 1.5 2 ; 5.8 2.7 4.1 1.0 2 ; 6.2 2.2 4.5 1.5 2 ;
         5.6 2.5 3.9 1.1 2 ; 5.9 3.2 4.8 1.8 2 ; 6.1 2.8 4.0 1.3 2 ;
         6.3 2.5 4.9 1.5 2 ; 6.1 2.8 4.7 1.2 2 ; 6.4 2.9 4.3 1.3 2 ;
         6.6 3.0 4.4 1.4 2 ; 6.8 2.8 4.8 1.4 2 ; 6.7 3.0 5.0 1.7 2 ;
         6.0 2.9 4.5 1.5 2 ; 5.7 2.6 3.5 1.0 2 ; 5.5 2.4 3.8 1.1 2 ;
         5.5 2.4 3.7 1.0 2 ; 5.8 2.7 3.9 1.2 2 ; 6.0 2.7 5.1 1.6 2 ;
         5.4 3.0 4.5 1.5 2 ; 6.0 3.4 4.5 1.6 2 ; 6.7 3.1 4.7 1.5 2 ;
         6.3 2.3 4.4 1.3 2 ; 5.6 3.0 4.1 1.3 2 ; 5.5 2.5 4.0 1.3 2 ;
         5.5 2.6 4.4 1.2 2 ; 6.1 3.0 4.6 1.4 2 ; 5.8 2.6 4.0 1.2 2 ;
         5.0 2.3 3.3 1.0 2 ; 5.6 2.7 4.2 1.3 2 ; 5.7 3.0 4.2 1.2 2 ;
         5.7 2.9 4.2 1.3 2 ; 6.2 2.9 4.3 1.3 2 ; 5.1 2.5 3.0 1.1 2 ;
         5.7 2.8 4.1 1.3 2 ; 6.3 3.3 6.0 2.5 3 ; 5.8 2.7 5.1 1.9 3 ;
         7.1 3.0 5.9 2.1 3 ; 6.3 2.9 5.6 1.8 3 ; 6.5 3.0 5.8 2.2 3 ;
         7.6 3.0 6.6 2.1 3 ; 4.9 2.5 4.5 1.7 3 ; 7.3 2.9 6.3 1.8 3 ;
         6.7 2.5 5.8 1.8 3 ; 7.2 3.6 6.1 2.5 3 ; 6.5 3.2 5.1 2.0 3 ;
         6.4 2.7 5.3 1.9 3 ; 6.8 3.0 5.5 2.1 3 ; 5.7 2.5 5.0 2.0 3 ;
         5.8 2.8 5.1 2.4 3 ; 6.4 3.2 5.3 2.3 3 ; 6.5 3.0 5.5 1.8 3 ;
         7.7 3.8 6.7 2.2 3 ; 7.7 2.6 6.9 2.3 3 ; 6.0 2.2 5.0 1.5 3 ;
         6.9 3.2 5.7 2.3 3 ; 5.6 2.8 4.9 2.0 3 ; 7.7 2.8 6.7 2.0 3 ;
         6.3 2.7 4.9 1.8 3 ; 6.7 3.3 5.7 2.1 3 ; 7.2 3.2 6.0 1.8 3 ;
         6.2 2.8 4.8 1.8 3 ; 6.1 3.0 4.9 1.8 3 ; 6.4 2.8 5.6 2.1 3 ;
         7.2 3.0 5.8 1.6 3 ; 7.4 2.8 6.1 1.9 3 ; 7.9 3.8 6.4 2.0 3 ;
         6.4 2.8 5.6 2.2 3 ; 6.3 2.8 5.1 1.5 3 ; 6.1 2.6 5.6 1.4 3 ;
         7.7 3.0 6.1 2.3 3 ; 6.3 3.4 5.6 2.4 3 ; 6.4 3.1 5.5 1.8 3 ;
         6.0 3.0 4.8 1.8 3 ; 6.9 3.1 5.4 2.1 3 ; 6.7 3.1 5.6 2.4 3 ;
         6.9 3.1 5.1 2.3 3 ; 5.8 2.7 5.1 1.9 3 ; 6.8 3.2 5.9 2.3 3 ;
         6.7 3.3 5.7 2.5 3 ; 6.7 3.0 5.2 2.3 3 ; 6.3 2.5 5.0 1.9 3 ;
         6.5 3.0 5.2 2.0 3 ; 6.2 3.4 5.4 2.3 3 ; 5.9 3.0 5.1 1.8 3 ];
          
% process training data

[x,idx] = unique(iris(:,1:4), 'rows');
t = iris(idx,5);
n = size(x, 1);

% modify options and perform Sammon mapping 

opts                = sammon;
opts.Display        = 'iter';
opts.TolFun         = 1e-9;
opts.Initialisation = 'pca';

% before = 69.007789
% after  =

tic
for i=1:20
[y, E] = sammon(x, 2, opts);
end
toc

% plot results

figure(1);
clf;
set(axes, 'FontSize', 16);
plot(y(t == 1,1), y(t == 1,2), 'r.', ...
     y(t == 2,1), y(t == 2,2), 'b+', ...
     y(t == 3,1), y(t == 3,2), 'go');
title(['Sammon Mapping of the Iris Dataset (stress = ' num2str(E) ')']);
legend('Setosa', 'Versicolour', 'Virginica');

% bye bye...

