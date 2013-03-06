/*
 * Copyright (C) 2009-2010 Andre Schulz, Florian Jung, Sebastian Hartte,
 *						   Daniel Trick, Christan Wojek, Konrad Schindler,
 *						   Jens Ackermann, Michael Goesele
 * Copyright (C) 2008-2009 Christopher Evans <chris.evans@irisys.co.uk>, MSc University of Bristol
 *
 * This file is part of SURFGPU.
 *
 * SURFGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SURFGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SURFGPU.  If not, see <http://www.gnu.org/licenses/>.
 */

//--S [] 2013/03/06: Sang-Wook Lee
//#include "ipoint.h"
#include "ipointGPU.h"
//--E [] 2013/03/06: Sang-Wook Lee

#include <vector>
#include <time.h>
#include <stdlib.h>

//-----------------------------------------------------------
// Kmeans clustering class (under development)
//  - Can be used to cluster points based on their location.
//  - Create Kmeans object and call Run with IpVec.
//  - Planned improvements include clustering based on motion 
//    and descriptor components.
//-----------------------------------------------------------

//--S [] 2013/03/06: Sang-Wook Lee
namespace surfgpu {
//--E [] 2013/03/06: Sang-Wook Lee

class Kmeans {

public:

  //! Destructor
  ~Kmeans() {};

  //! Constructor
  Kmeans() {};

  //! Do it all!
  void Run(IpVec *ipts, int clusters, bool init = false);

  //! Set the ipts to be used
  void SetIpoints(IpVec *ipts);

  //! Randomly distribute 'n' clusters
  void InitRandomClusters(int n);

  //! Assign Ipoints to clusters
  bool AssignToClusters();

  //! Calculate new cluster centers
  void RepositionClusters();

  //! Function to measure the distance between 2 ipoints
  float Distance(Ipoint &ip1, Ipoint &ip2);

  //! Vector stores ipoints for this run
  IpVec *ipts;

  //! Vector stores cluster centers
  IpVec clusters;

};

//-------------------------------------------------------

void Kmeans::Run(IpVec *ipts, int clusters, bool init)
{
  if (!ipts->size()) return;

  SetIpoints(ipts);

  if (init) InitRandomClusters(clusters);
  
  while (AssignToClusters());
  {
    RepositionClusters();
  }
}

//-------------------------------------------------------

void Kmeans::SetIpoints(IpVec *ipts)
{
  this->ipts = ipts;
}

//-------------------------------------------------------

void Kmeans::InitRandomClusters(int n)
{
  // clear the cluster vector
  clusters.clear();

  // Seed the random number generator
  srand((int)time(NULL));

  // add 'n' random ipoints to clusters list as initial centers
  for (int i = 0; i < n; ++i)
  {
    clusters.push_back(ipts->at(rand() % ipts->size()));
  }
}

//-------------------------------------------------------

bool Kmeans::AssignToClusters()
{
  bool Updated = false;

  // loop over all Ipoints and assign each to closest cluster
  for (unsigned int i = 0; i < ipts->size(); ++i)
  {
    float bestDist = FLT_MAX;
    int oldIndex = ipts->at(i).clusterIndex;

    for (unsigned int j = 0; j < clusters.size(); ++j)
    {
      float currentDist = Distance(ipts->at(i), clusters[j]);
      if (currentDist < bestDist)
      {
        bestDist = currentDist;
        ipts->at(i).clusterIndex = j;
      }
    }

    // determine whether point has changed cluster
    if (ipts->at(i).clusterIndex != oldIndex) Updated = true;
  }

  return Updated;
}

//-------------------------------------------------------

void Kmeans::RepositionClusters()
{
  float x, y, dx, dy, count;

  for (unsigned int i = 0; i < clusters.size(); ++i)
  {
    x = y = dx = dy = 0;
    count = 1;

    for (unsigned int j = 0; j < ipts->size(); ++j)
    {
      if (ipts->at(j).clusterIndex == i)
      {
        Ipoint ip = ipts->at(j);
        x += ip.x;
        y += ip.y;
        dx += ip.dx;
        dy += ip.dy;
        ++count;
      }
    }

    clusters[i].x = x/count;
    clusters[i].y = y/count;
    clusters[i].dx = dx/count;
    clusters[i].dy = dy/count;
  }
}

//-------------------------------------------------------

float Kmeans::Distance(Ipoint &ip1, Ipoint &ip2)
{
  return sqrt(pow(ip1.x - ip2.x, 2) 
            + pow(ip1.y - ip2.y, 2)
            /*+ pow(ip1.dx - ip2.dx, 2) 
            + pow(ip1.dy - ip2.dy, 2)*/);
}

//-------------------------------------------------------

//--S [] 2013/03/06: Sang-Wook Lee
}  // namespace surfgpu
//--E [] 2013/03/06: Sang-Wook Lee
