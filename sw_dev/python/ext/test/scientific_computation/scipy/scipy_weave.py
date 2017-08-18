import scipy.weave
import numpy as np

#%%-------------------------------------------------------------------
weave.inline('std::cout << a << std::endl;', ['a'])
a = 'string'
weave.inline('std::cout << a << std::endl;', ['a'])

sum = np.zeros(3, dtype=np.uint8)
code = """
	for (int i = 0; i < 10; ++i)
		sum[0] += i;
	for (int i = 0; i < 100; ++i)
		sum[1] += i;
	for (int i = 0; i < 100; ++i)
		sum[2] += i;
"""
weave.inline(code, ["sum"])
