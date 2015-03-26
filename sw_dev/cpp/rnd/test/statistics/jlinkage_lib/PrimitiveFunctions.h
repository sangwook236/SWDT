#ifndef PRIMITIVEFUNCTIONS_H
#define PRIMITIVEFUNCTIONS_H

enum MODELTYPE{
	MT_PLANE,
	MT_LINE,
	MT_VP,
	MT_SIZE
};

// given two vector of the same size compute euclidean distance
template<typename T>
T VecEuclideanDist(const std::vector<T> &nVecI, const std::vector<T> &nVecJ){
	assert(nVecI.size() == nVecJ.size());
	T eucDist = 0.0f;
	std::vector<T> tempVec = std::vector<T>(nVecI.size());
	for(unsigned int i=0; i<nVecI.size(); i++)
		eucDist += (nVecI[i] - nVecJ[i]) * (nVecI[i] - nVecJ[i]);

	return sqrt(eucDist);
}

inline float PtPairDistance_Euclidean(const std::vector<float> &nVecI, const std::vector<float> &nVecJ)
{
	return VecEuclideanDist(nVecI,nVecJ);
}

//////////////////////////////////////////////////////////////////////////
//      Plane
//////////////////////////////////////////////////////////////////////////

inline std::vector<float>  *GetFunction_Plane(const std::vector<sPt *> &nDataPtXMss, const std::vector<unsigned int>  &nSelectedPts){

		std::vector<float>   *nReturningVector = new std::vector<float> (4,0.0f); // normal and d
		
		// normal found as cross product of X2 - X1, X3 - X1
		float x1 = (*nDataPtXMss[nSelectedPts[1]]->mCoord)[0] - (*nDataPtXMss[nSelectedPts[0]]->mCoord)[0];
		float x2 = (*nDataPtXMss[nSelectedPts[2]]->mCoord)[0] - (*nDataPtXMss[nSelectedPts[0]]->mCoord)[0];
		float y1 = (*nDataPtXMss[nSelectedPts[1]]->mCoord)[1] - (*nDataPtXMss[nSelectedPts[0]]->mCoord)[1];
		float y2 = (*nDataPtXMss[nSelectedPts[2]]->mCoord)[1] - (*nDataPtXMss[nSelectedPts[0]]->mCoord)[1];
		float z1 = (*nDataPtXMss[nSelectedPts[1]]->mCoord)[2] - (*nDataPtXMss[nSelectedPts[0]]->mCoord)[2];
		float z2 = (*nDataPtXMss[nSelectedPts[2]]->mCoord)[2] - (*nDataPtXMss[nSelectedPts[0]]->mCoord)[2];

		(*nReturningVector)[0] = y1 * z2 - z1 * y2;
		(*nReturningVector)[1] = z1 * x2 - x1 * z2;
		(*nReturningVector)[2] = x1 * y2 - y1 * x2;
		// compute d
		(*nReturningVector)[3] =  - ((*nReturningVector)[0] * (*nDataPtXMss[nSelectedPts[0]]->mCoord)[0]) 
									 - ((*nReturningVector)[1] * (*nDataPtXMss[nSelectedPts[0]]->mCoord)[1]) 
									 - ((*nReturningVector)[2] * (*nDataPtXMss[nSelectedPts[0]]->mCoord)[2]); 
		
		// Normalize
		float normN = sqrt((*nReturningVector)[0] * (*nReturningVector)[0] + (*nReturningVector)[1] * (*nReturningVector)[1] +  (*nReturningVector)[2] * (*nReturningVector)[2]);
		(*nReturningVector)[0]  = (*nReturningVector)[0] /normN;
		(*nReturningVector)[1]  = (*nReturningVector)[1] /normN;
		(*nReturningVector)[2]  = (*nReturningVector)[2] /normN;
		(*nReturningVector)[3]  = (*nReturningVector)[3] /normN;

		return nReturningVector;
	}

inline float DistanceFunction_Plane(const std::vector<float> &nModel, const std::vector<float>  &nDataPt){
	return fabs(nModel[0] * nDataPt[0] + nModel[1] * nDataPt[1] + nModel[2] * nDataPt[2] + nModel[3]);
}

//////////////////////////////////////////////////////////////////////////
//       Line
//////////////////////////////////////////////////////////////////////////

inline std::vector<float>  *GetFunction_Line(const std::vector<sPt *> &nDataPtXMss, const std::vector<unsigned int>  &nSelectedPts){
	
	std::vector<float>   *nReturningVector = new std::vector<float> (3,0.0f); // normal and d
	
	float x1 = (*nDataPtXMss[nSelectedPts[0]]->mCoord)[0];
	float x2 = (*nDataPtXMss[nSelectedPts[1]]->mCoord)[0];
	float y1 = (*nDataPtXMss[nSelectedPts[0]]->mCoord)[1];
	float y2 = (*nDataPtXMss[nSelectedPts[1]]->mCoord)[1];

	// A
	(*nReturningVector)[0] = (y2 - y1) / (y2*x1 - x2*y1);

	// C
	(*nReturningVector)[2] = -1;

	// B
	(*nReturningVector)[1] = (-(*nReturningVector)[2] -  (*nReturningVector)[0] * x1)/y1;

	return nReturningVector;
}

inline float DistanceFunction_Line(const std::vector<float> &nModel, const std::vector<float>  &nDataPt){
	//dist(i) = sqrt( (P(1:2)'*X(:,i)+P(3) ).^2  / (P(1)^2+P(2)^2));
	float dist = fabs(nModel[0] * nDataPt[0] + nModel[1] * nDataPt[1] + nModel[2]) / sqrt(nModel[0] * nModel[0] + nModel[1] * nModel[1]);
	return  dist;
}

//////////////////////////////////////////////////////////////////////////
//        Vanishing Point
//                       add by FC
//////////////////////////////////////////////////////////////////////////

inline void vec_cross(float a1, float b1, float c1,
							 float a2, float b2, float c2,
							 float& a3, float& b3, float& c3)
{
	a3 = b1*c2 - c1*b2;
	b3 = -(a1*c2 - c1*a2);
	c3 = a1*b2 - b1*a2;
}

inline void vec_norm(float& a, float& b, float& c)
{
	float len = sqrt(a*a+b*b+c*c);
	a/=len; b/=len; c/=len;
}

inline std::vector<float>  *GetFunction_VP(const std::vector<sPt *> &nDataPtXMss, const std::vector<unsigned int>  &nSelectedPts){

	std::vector<float>   *nReturningVector = new std::vector<float> (3,0.0f);

	float xs0 = (*nDataPtXMss[nSelectedPts[0]]->mCoord)[0];
	float ys0 = (*nDataPtXMss[nSelectedPts[0]]->mCoord)[1];
	float xe0 = (*nDataPtXMss[nSelectedPts[0]]->mCoord)[2];
	float ye0 = (*nDataPtXMss[nSelectedPts[0]]->mCoord)[3];
	float xs1 = (*nDataPtXMss[nSelectedPts[1]]->mCoord)[0];
	float ys1 = (*nDataPtXMss[nSelectedPts[1]]->mCoord)[1];
	float xe1 = (*nDataPtXMss[nSelectedPts[1]]->mCoord)[2];
	float ye1 = (*nDataPtXMss[nSelectedPts[1]]->mCoord)[3];

	float l0[3],l1[3],v[3];
	vec_cross(xs0,ys0,1,
		xe0,ye0,1,
		l0[0],l0[1],l0[2]);
	vec_cross(xs1,ys1,1,
		xe1,ye1,1,
		l1[0],l1[1],l1[2]);
	vec_cross(l0[0],l0[1],l0[2],
		l1[0],l1[1],l1[2],
		v[0],v[1],v[2]);
	vec_norm(v[0],v[1],v[2]);

	(*nReturningVector)[0] = v[0];
	(*nReturningVector)[1] = v[1];
	(*nReturningVector)[2] = v[2];

	return nReturningVector;
}

inline float DistanceFunction_VP(const std::vector<float> &nModel, const std::vector<float>  &nDataPt){
	float l[3], mid[3] = {(nDataPt[0]+nDataPt[2])/2.0, (nDataPt[1]+nDataPt[3])/2.0, 1};
	vec_cross(mid[0],mid[1],mid[2],
		nModel[0],nModel[1],nModel[2],
		l[0],l[1],l[2]);
	float dist = fabs(l[0]*nDataPt[0]+l[1]*nDataPt[1]+l[2]) / sqrt(l[0]*l[0]+l[1]*l[1]);
	return  dist;
}

inline float PtPairDistance_VP(const std::vector<float> &nVecI, const std::vector<float> &nVecJ)
{
	std::vector<float> midI(2), midJ(2);
	midI[0] = (nVecI[0]+nVecI[2])/2.0;
	midI[1] = (nVecI[1]+nVecI[3])/2.0;
	midJ[0] = (nVecJ[0]+nVecJ[2])/2.0;
	midJ[1] = (nVecJ[1]+nVecJ[3])/2.0;
	return PtPairDistance_Euclidean(midI,midJ);
}

#endif