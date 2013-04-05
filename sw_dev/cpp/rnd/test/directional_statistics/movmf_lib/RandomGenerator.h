// RandomGenerator.h: interface for the random number generator classes.
//		These classes are to generate random numbers needed by I/O and
//		BSG floorplan algorithm implementation including simulated
//		annealing.
// Modified from GNU Scientific Library 0.4.1
// The GNU Scientific Library can be downloaded from:
//		ftp://sourceware.cygnus.com/pub/gsl
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_RANDOMGENERATOR_H__DE1C1AFF_C9AF_4790_B7F6_F5B9B840789F__INCLUDED_)
#define AFX_RANDOMGENERATOR_H__DE1C1AFF_C9AF_4790_B7F6_F5B9B840789F__INCLUDED_

#include <cstddef>

#define DEFAULT_SEED 0
//////////////////////////////////////////////////////////////////////
// base class of random number generators
class RandomGenerator  
{
protected:
    const char *name;			// name of the generator
    unsigned long int max;		// maximum value of the random number
    unsigned long int min;		// minimum value of the random number
    size_t size;			// the size of state ?? do not have much use

public:
	RandomGenerator();
	virtual ~RandomGenerator();

	inline float GetUniformPos();			/* return a positive double precision floating point
												number uniformly distributed in (0, 1) */
	unsigned long int GetUniformInt(unsigned long int n);
											/* return a random integer from 0 to n-1 inclusive,
												all integers in [0, n-1] are equally likely */
	
	// Gaussian distribution
	float GetGaussian(const float sigma); /* return a gaussian random number, with mean zero and
												standard deviation sigma */
	float GetGaussianPDF(const float x, const float sigma);
											/* compute the probability density at x for a gaussian
												distribution with standard deviation sigma */
	inline float GetUGaussian() { return GetGaussian(1.0); }
											/* return a gaussian random number with mean zero and
												deviation 1.0 */

	// Get properties of the random number generator
	inline unsigned long int GetMax() { return max; };
	inline unsigned long int GetMin() { return min; };
	inline const char* GetName() { return name; };
	inline size_t GetSize() { return size; };

	// functions that will be overrided
    virtual void Set(unsigned long int seed) = 0;		/* initialize the generator */
    virtual unsigned long int Get() = 0;				/* return a random integer value, all integers
															in [min, max] are equally likely */
    virtual float GetUniform() = 0;					/* return a double precision random floating point
															number uniformly distributed in [0, 1) */
};

//////////////////////////////////////////////////////////////////////
// derived classes

// MT19937 generator (simulation quality)
/* "Mersenne Twister" generator by Makoto Matsumoto and Takuji Nishimura

   Makoto Matsumoto has a web page with more information about the
   generator, http://www.math.keio.ac.jp/~matumoto/emt.html. 

   The paper below has details of the algorithm.

   From: Makoto Matsumoto and Takuji Nishimura, "Mersenne Twister: A
   623-dimensionally equidistributerd uniform pseudorandom number
   generator". ACM Transactions on Modeling and Computer Simulation,
   Vol. 8, No. 1 (Jan. 1998), Pages 3-30

   You can obtain the paper directly from Makoto Matsumoto's web page.

   The period of this generator is 2^{19937} - 1.
*/

class RandomGenerator_MT19937 : public RandomGenerator
{

#define MT_N 624	/* Period parameters */
#define MT_M 397

protected:
	/* most significant w-r bits */
	const unsigned long UPPER_MASK;

	/* least significant r bits */
	const unsigned long LOWER_MASK;

	int mti;						//state
	unsigned long mt[MT_N];			//state

public:
	RandomGenerator_MT19937();
	virtual ~RandomGenerator_MT19937();

	virtual void Set(unsigned long int seed);
	inline virtual unsigned long int Get();
	virtual float GetUniform();
};

// Tausworthe generator (simulation quality)
/* The period of this generator is about 2^88.

   From: P. L'Ecuyer, "Maximally Equidistributed Combined Tausworthe
   Generators", Mathematics of Computation, 65, 213 (1996), 203--213.

   This is available on the net from L'Ecuyer's home page,

   http://www.iro.umontreal.ca/~lecuyer/myftp/papers/tausme.ps
   ftp://ftp.iro.umontreal.ca/pub/simulation/lecuyer/papers/tausme.ps
*/

class RandomGenerator_Taus : public RandomGenerator
{
protected:
	unsigned long int s1, s2, s3;		// state

public:
	RandomGenerator_Taus();
	virtual ~RandomGenerator_Taus();

	virtual void Set(unsigned long int seed);
	inline virtual unsigned long int Get();
	virtual float GetUniform();
};

// TT800 generator
/* This is the TT800 twisted GSFR generator for 32 bit integers. It
   has been superceded by MT19937 (mt.c). The period is 2^800.

   This implementation is based on tt800.c, July 8th 1996 version by
   M. Matsumoto, email: matumoto@math.keio.ac.jp

   From: Makoto Matsumoto and Yoshiharu Kurita, "Twisted GFSR
   Generators II", ACM Transactions on Modelling and Computer
   Simulation, Vol. 4, No. 3, 1994, pages 254-266.
*/

class RandomGenerator_TT800 : public RandomGenerator
{

#define TT_N 25
#define TT_M 7

protected:
    int n;						//state
    unsigned long int x[TT_N];		//state

public:
	RandomGenerator_TT800();
	virtual~RandomGenerator_TT800();

	virtual void Set(unsigned long int seed);
	inline virtual unsigned long int Get();
	virtual float GetUniform();
};

// R250 generator
/* This is a shift-register random number generator.

   The period of this generator is about 2^250.

   The algorithm works for any number of bits. It is implemented here
   for 32 bits.

   From: S. Kirkpatrick and E. Stoll, "A very fast shift-register
   sequence random number generator", Journal of Computational Physics,
   40, 517-526 (1981).
*/

class RandomGenerator_R250 : public RandomGenerator
{
protected:
    int i;					//state
    unsigned long x[250];	//state

public:
	RandomGenerator_R250();
	virtual ~RandomGenerator_R250();

	virtual void Set(unsigned long int seed);
	inline virtual unsigned long int Get();
	virtual float GetUniform();
};

#endif // !defined(AFX_RANDOMGENERATOR_H__DE1C1AFF_C9AF_4790_B7F6_F5B9B840789F__INCLUDED_)
