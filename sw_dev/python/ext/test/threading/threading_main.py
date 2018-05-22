#!/usr/bin/env python

# REF [sit] >> https://docs.python.org/3/tutorial/stdlib2.html#multi-threading
# REF [site] >> https://docs.python.org/3.4/library/threading.html

import threading
import zipfile

class AsyncZip(threading.Thread):
	def __init__(self, infile, outfile):
		super().__init__()
		self.infile = infile
		self.outfile = outfile

	def run(self):
		f = zipfile.ZipFile(self.outfile, 'w', zipfile.ZIP_DEFLATED)
		f.write(self.infile)
		f.close()
		print('Finished background zip of:', self.infile)

def main():
	background = AsyncZip('..', 'myarchive.zip')
	background.start()
	print('The main program continues to run in foreground.')

	background.join()  # Wait for the background task to finish.
	print('Main program waited until background was done.')

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
