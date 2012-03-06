clc;
clear all;
% /************************************************************************
%  * File Name:     getViterbi.m
%  * Programmer:    Bormi Shin (bshin@andrew.cmu.edu)
%  * Category:      Computationl Biology/Bioinformatics
%  * Description:   General implementation of matlab version of Viterbi 
%  *                algorithm specifically written for gene structure 
%  *                finding problem in mind. However, it can be modified 
%  *                to suit the goal of a user.
%  * Input:         Transition Probability Matrix
%  *                Emission Probability Matrix
%  *                Initial Probability Matrix
%  *                States (ie. Exon,Intron, Intergenic, etc.)
%  *                Sequence File Name <-- Optional
%  * Output:        Most probable state sequence ('Path') into output.txt
%  * Further Detail: http://en.wikipedia.org/wiki/Viterbi_algorithm
%  * Last Modified: 05/08/2006
%  * Known Bugs: None as long as valid parameters are passed
%  * Compiler: Matlab 7.0+ 
%  ***********************************************************************/
function getViterbi(Ptransition, Pemission, Pinitial, S, SeqTxt)
% Read in sequence file and store it as a character array
% SeqTxt can be replaced by your 'file name.txt'. In that case, you can 
% remove SeqTxt paramter from the function.
S=[];
fid = fopen(SeqTxt, 'r');
while feof(fid) == 0
    A = fgetl(fid);
    S=[S,A];
end
fclose(fid);
S1 = char(S);
% Initialize states and transition, emission, and initial probabilities
Pa=Ptransition;
Pb=Pemission';
Pi=Pinitial;
States = S;
% Determin column and row size
Col=length(States);
Row=length(S1);
% Initialize delta and psi variables 
delta = zeros(Row,Col);
psi = zeros(Row-1,Col);
% Initialization
t = 1;
c=1;
switch upper(S1(t,1))
    case 'A'
        c=1;
    case 'C' 
        c=2;
    case 'G' 
        c=3;
    case 'T' 
        c=4;
    otherwise
        r= rand(1);
        if(r<=0.25)
            c=1;
        elseif(0.25<r & r<=0.5)
            c=2;
        elseif(0.5<r & r<=0.75)
            c=3;
        elseif(0.75<r)
            c=4;
        end
end
for i = 1:Col
    if(Pi(i) == 0) temp1 = -inf; else temp1 = double(log(Pi(i))); end
    if(Pb(c,i) == 0) temp2 = -inf; else temp2 = double(log(Pb(c,i))); end
    delta(t,i) = temp1+temp2;
end

% Recursion
for t = 2:Row     
    for j = 1:Col    
        p = -inf;
        k = 0; 
        for i = 1:Col
            %if(delta(t-1,i) == 0) temp1 = -inf; else temp1 = double(log(delta(t-1,i))); end
            if(Pa(i,j) == 0) temp2 = -inf; else temp2 = double(log(Pa(i,j))); end
            q = delta(t-1,i)+temp2;
            if ( q >= p )        
                p = q;
                k = i;
            end
        end
        switch upper(S1(1,j))
            case 'A'
                c=1;
            case 'C' 
                c=2;
            case 'G' 
                c=3;
            case 'T' 
                c=4;   
            otherwise
                r= rand(1);
                if(r<=0.25)
                    c=1;
                elseif(0.25<r & r<=0.5)
                    c=2;
                elseif(0.5<r & r<=0.75)
                    c=3;
                elseif(0.75<r)
                    c=4;
                end
        end
        if(Pb(c,j) == 0) temp3 = -inf; else temp3 = double(log(Pb(c,j))); end
        delta(t,j) = p+ temp3; 
        psi(t,j) = k;
    end                  
end
% Termination
  t = Row;
  p = -inf;
  k = 0; 
  for i = 1:Col
    q = delta(t,i);
    if ( q >= p )  
      p = q;
      k = i;
    end
  end
  StateIdxs=zeros(1,Row);
  StateIdxs(t) = k;
  % Path (state sequence) backtracking
  for t = Row-1:-1:1  
      StateIdxs(t) = psi(t+1, StateIdxs(t+1));
  end
  %StateIdxs
  %dlmwrite('States.txt', StateIdx, '\n');
  fout=fopen('output.txt', 'w');
  for i=1:ceil(length(StateIdxs)/50)
    fprintf(fout,'%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d \n',StateIdxs(1,i:(i+49)));
  end
  fclose(fout);