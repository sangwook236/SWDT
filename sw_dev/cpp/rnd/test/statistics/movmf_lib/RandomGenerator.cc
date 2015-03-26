// RandomGenerator.cpp: implementation of the random number generator classes.
//
//////////////////////////////////////////////////////////////////////

#include <math.h>
#include "RandomGenerator.h"

#define PI 3.14159265358979323846264338328

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

RandomGenerator::RandomGenerator()
{

}

RandomGenerator::~RandomGenerator()
{

}

// Implementation of base class

float RandomGenerator::GetUniformPos()
{
  float x ;
  do
    {
      x = GetUniform();
    }
  while (x == 0) ;

  return x ;
}

unsigned long int RandomGenerator::GetUniformInt(unsigned long int n)
{
  unsigned long int offset = min;
  unsigned long int range = max - offset;
  unsigned long int scale = range / n;
  unsigned long int k;

  if (n > range) 
		n=range;

  do
    {
      k = (Get() - offset) / scale;
    }
  while (k >= n);

  return k;
}

float RandomGenerator::GetGaussian(const float sigma)
{
#if 1 /* Polar (Box-Mueller) method; See Knuth v2, 3rd ed, p122 */
  float x, y, r2;
  
  do
    {
      /* choose x,y in uniform square (-1,-1) to (+1,+1) */

      x = -1 + 2 * GetUniform();
      y = -1 + 2 * GetUniform();

      /* see if it is in the unit circle */
      r2 = x * x + y * y;
    }
  while (r2 > 1.0 || r2 == 0);

  /* Box-Muller transform */
  return sigma * y * sqrt (-2.0 * log (r2) / r2); /* only one random deviate is produced, the
														other is discarded. Because saving the
														other in a static variable would screw up
														re-entrant or theaded code. */
#endif
#if 0 /* Ratio method (Kinderman-Monahan); see Knuth v2, 3rd ed, p130 */
      /* K+M, ACM Trans Math Software 3 (1977) 257-260. */
  float u,v,x,xx;

  do {
      v = GetUniform(r);
      do {
          u = GetUniform(r);
      } 
      while (u==0);
      /* Const 1.715... = sqrt(8/e) */
      x = 1.71552776992141359295*(v-0.5)/u;
  }
  while (x*x > -4.0*log(u));
  return sigma*x;
#endif
}

float RandomGenerator::GetGaussianPDF(const float x, const float sigma)
{
  float u = x / fabs(sigma) ;
  float p = (1 / (sqrt (2 * PI) * fabs(sigma)) ) * exp (-u * u / 2);
  return p;
}

//////////////////////////////////////////////////////////////////////
// Implementation of MT_19937 generator
//////////////////////////////////////////////////////////////////////

RandomGenerator_MT19937::RandomGenerator_MT19937() : UPPER_MASK(0x80000000UL), LOWER_MASK(0x7fffffffUL)
{
	name="mt19937";	//??
	max=0xffffffffUL;			/* RAND_MAX  */
	min=0;						/* RAND_MIN  */
	size=sizeof(mti)+sizeof(mt);	// ??
}

RandomGenerator_MT19937::~RandomGenerator_MT19937()
{
}

unsigned long RandomGenerator_MT19937::Get()
{
  unsigned long k ;

#define MAGIC(y) (((y)&0x1) ? 0x9908b0dfUL : 0)

  if (mti >= MT_N)
    {	/* generate N words at one time */
      int kk;

      for (kk = 0; kk < MT_N - MT_M; kk++)
	{
	  unsigned long y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
	  mt[kk] = mt[kk + MT_M] ^ (y >> 1) ^ MAGIC(y);
	}
      for (; kk < MT_N - 1; kk++)
	{
	  unsigned long y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
	  mt[kk] = mt[kk + (MT_M - MT_N)] ^ (y >> 1) ^ MAGIC(y);
	}

      {
	unsigned long y = (mt[MT_N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
	mt[MT_N - 1] = mt[MT_M - 1] ^ (y >> 1) ^ MAGIC(y);
      }

      mti = 0;
    }

  /* Tempering */
  
  k = mt[mti];
  k ^= (k >> 11);
  k ^= (k << 7) & 0x9d2c5680UL;
  k ^= (k << 15) & 0xefc60000UL;
  k ^= (k >> 18);

  mti++;

  return k;
}

float RandomGenerator_MT19937::GetUniform()
{
  return Get() / 4294967296.0 ;
}

void RandomGenerator_MT19937::Set(unsigned long int seed)
{
  int i;

  if (seed == 0)
    seed = 4357;	/* the default seed is 4357 */

  mt[0] = seed & 0xffffffffUL;

  /* We use the congruence s_{n+1} = (69069*s_n) mod 2^32 to
     initialize the state. This works because ANSI-C unsigned long
     integer arithmetic is automatically modulo 2^32 (or a higher
     power of two), so we can safely ignore overflow. */

#define LCG(n) ((69069 * n) & 0xffffffffUL)

  for (i = 1; i < MT_N; i++)
    mt[i] = LCG (mt[i - 1]);

  mti = i;
}


//////////////////////////////////////////////////////////////////////
// Implementation of Taus generator
//////////////////////////////////////////////////////////////////////

RandomGenerator_Taus::RandomGenerator_Taus()
{
	name="taus";			/* name */
	max=0xffffffffUL;		/* RAND_MAX */
	min=0;			        /* RAND_MIN */
	size=sizeof(unsigned long int)*3;
}

RandomGenerator_Taus::~RandomGenerator_Taus()
{
}

unsigned long int RandomGenerator_Taus::Get()
{
#define MASK 0xffffffffUL
#define TAUSWORTHE(s,a,b,c,d) (((s &c) <<d) &MASK) ^ ((((s <<a) &MASK)^s) >>b)

  s1 = TAUSWORTHE (s1, 13, 19, 4294967294UL, 12);
  s2 = TAUSWORTHE (s2, 2, 25, 4294967288UL, 4);
  s3 = TAUSWORTHE (s3, 3, 11, 4294967280UL, 17);

  return (s1 ^ s2 ^ s3);
}

float RandomGenerator_Taus::GetUniform()
{
  return Get() / 4294967296.0 ;
}

void RandomGenerator_Taus::Set(unsigned long seed)
{
  if (seed == 0)
    seed = 1;	/* default seed is 1 */

#define LCG(n) ((69069 * n) & 0xffffffffUL)
  s1 = LCG (seed);
  s2 = LCG (s1);
  s3 = LCG (s2);

  /* "warm it up" */
  Get();
  Get();
  Get();
  Get();
  Get();
  Get();
  return;
}


//////////////////////////////////////////////////////////////////////
// Implementation of TT800 generator
//////////////////////////////////////////////////////////////////////

RandomGenerator_TT800::RandomGenerator_TT800()
{
	name="tt800";			/* name */
	max=0xffffffffUL;		/* RAND_MAX */
	min=0;			        /* RAND_MIN */
	size=sizeof(n)+sizeof(x);
}

RandomGenerator_TT800::~RandomGenerator_TT800()
{
}

unsigned long int RandomGenerator_TT800::Get()
{
  /* this is the magic vector, a */

  const unsigned long mag01[2] =
  {0x00000000, 0x8ebfd028UL};
  unsigned long int y;

  if (n >= TT_N)
    {
      int i;
      for (i = 0; i < TT_N - TT_M; i++)
	{
	  x[i] = x[i + TT_M] ^ (x[i] >> 1) ^ mag01[x[i] % 2];
	}
      for (; i < TT_N; i++)
	{
	  x[i] = x[i + (TT_M - TT_N)] ^ (x[i] >> 1) ^ mag01[x[i] % 2];
	};
      n = 0;
    }

  y = x[n];
  y ^= (y << 7) & 0x2b5b2500UL;		/* s and b, magic vectors */
  y ^= (y << 15) & 0xdb8b0000UL;	/* t and c, magic vectors */
  y &= 0xffffffffUL;	/* you may delete this line if word size = 32 */

  /* The following line was added by Makoto Matsumoto in the 1996
     version to improve lower bit's correlation.  Delete this line
     to use the code published in 1994.  */

  y ^= (y >> 16);	/* added to the 1994 version */

  n = n + 1;

  return y;
}

float RandomGenerator_TT800::GetUniform()
{
  return Get() / 4294967296.0 ;
}

void RandomGenerator_TT800::Set(unsigned long int seed)
{
  const int init_n = 0;
  const unsigned long int init_x[TT_N] =
		{0x95f24dabUL, 0x0b685215UL, 0xe76ccae7UL,
		 0xaf3ec239UL, 0x715fad23UL, 0x24a590adUL,
		 0x69e4b5efUL, 0xbf456141UL, 0x96bc1b7bUL,
		 0xa7bdf825UL, 0xc1de75b7UL, 0x8858a9c9UL,
		 0x2da87693UL, 0xb657f9ddUL, 0xffdc8a9fUL,
		 0x8121da71UL, 0x8b823ecbUL, 0x885d05f5UL,
		 0x4e20cd47UL, 0x5a9ad5d9UL, 0x512c0c03UL,
		 0xea857ccdUL, 0x4cc1d30fUL, 0x8891a8a1UL,
		 0xa6b7aadbUL};

  if (seed == 0)	/* default seed is given explicitly in the original code */
    {
      n = init_n;
	  for (int i=0; i<TT_N; i++) x[i] = init_x[i];
	}
  else
    {
      int i;

      n = 0;

      x[0] = seed & 0xffffffffUL;

      for (i = 1; i < TT_N; i++)
		x[i] = (69069 * x[i - 1]) & 0xffffffffUL;
    }

  return;
}


//////////////////////////////////////////////////////////////////////
// Implementation of R250 generator
//////////////////////////////////////////////////////////////////////

RandomGenerator_R250::RandomGenerator_R250()
{
	name="r250";			/* name */
	max=0xffffffffUL;		/* RAND_MAX */
	min=0;			        /* RAND_MIN */
	size=sizeof(i)+sizeof(x);
}

RandomGenerator_R250::~RandomGenerator_R250()
{
}

unsigned long int RandomGenerator_R250::Get()
{
  unsigned long int k;
  int j;

  if (i >= 147)
    {
      j = i - 147;
    }
  else
    {
      j = i + 103;
    }

  k = x[i] ^ x[j];
  x[i] = k;

  if (i >= 249)
    {
      i = 0;
    }
  else
    {
      i = i + 1;
    }

  return k;
}

float RandomGenerator_R250::GetUniform()
{
  return Get() /  4294967296.0 ;
}

void RandomGenerator_R250::Set(unsigned long int seed)
{
  int j;

  if (seed == 0)
    seed = 1;	/* default seed is 1 */

  i = 0;

#define LCG(n) ((69069 * n) & 0xffffffffUL)

  for (j = 0; j < 250; j++)	/* Fill the buffer  */
    {
      seed = LCG (seed);
      x[j] = seed;
    }

  {
    /* Masks for turning on the diagonal bit and turning off the
       leftmost bits */

    unsigned long int msb = 0x80000000UL;
    unsigned long int mask = 0xffffffffUL;

    for (j = 0; j < 32; j++)
      {
	int k = 7 * j + 3;	/* Select a word to operate on        */
	x[k] &= mask;	/* Turn off bits left of the diagonal */
	x[k] |= msb;	/* Turn on the diagonal bit           */
	mask >>= 1;
	msb >>= 1;
      }
  }

  return;
}
