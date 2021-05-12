#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os, re
import numpy as np
import jpype
import jpype.imports

# REF [site] >> https://github.com/lebedov/python-pdfbox/
def initialize_java_vm():
	import hashlib
	import pathlib
	import html.parser
	import urllib.request
	import appdirs
	import pkg_resources

	pdfbox_archive_url = 'https://archive.apache.org/dist/pdfbox/'

	class _PDFBoxVersionsParser(html.parser.HTMLParser):
		"""
		Class for parsing versions available on PDFBox archive site.
		"""

		def feed(self, data):
			self.result = []
			super(_PDFBoxVersionsParser, self).feed(data)

		def handle_starttag(self, tag, attrs):
			if tag == 'a':
				for a in attrs:
					if a[0] == 'href':
						s = a[1].strip('/')
						if re.search('\d+\.\d+\.\d+.*', s):
							self.result.append(s)

	def _verify_sha512(data, digest):
		"""
		Verify SHA512 checksum.
		"""

		return hashlib.sha512(data).hexdigest() == digest

	def _get_latest_pdfbox_url():
		r = urllib.request.urlopen(pdfbox_archive_url)
		try:
			data = r.read()
		except:
			raise RuntimeError('error retrieving %s' % pdfbox_archive_url)
		else:
			data = data.decode('utf-8')
		p = _PDFBoxVersionsParser()
		p.feed(data)

		# Temporarily disallow PDFBox 3 because of change in command line interface:
		versions = list(filter(lambda v: pkg_resources.parse_version(v).major<3, p.result))
		latest_version = sorted(versions, key=pkg_resources.parse_version)[-1]
		return pdfbox_archive_url + latest_version + '/pdfbox-app-' + latest_version + '.jar'

	def _get_pdfbox_path():
		"""
		Return path to local copy of PDFBox jar file.
		"""

		# Use PDFBOX environmental variable if it exists:
		if 'PDFBOX' in os.environ:
			pdfbox_path = pathlib.Path(os.environ['PDFBOX'])
			if not pdfbox_path.exists():
				raise RuntimeError('pdfbox not found')
			return pdfbox_path

		# Use platform-specific cache directory:
		a = appdirs.AppDirs('python-pdfbox')
		cache_dir = pathlib.Path(a.user_cache_dir)

		# Try to find pdfbox-app-*.jar file with most recent version in cache directory:
		file_list = list(cache_dir.glob('pdfbox-app-*.jar'))
		if file_list:
			def f(s):
				v = re.search('pdfbox-app-([\w\.\-]+)\.jar', s.name).group(1)
				return pkg_resources.parse_version(v)
			return sorted(file_list, key=f)[-1]
		else:
			# If no jar files are cached, find the latest version jar, retrieve it,
			# cache it, and verify its checksum:
			pdfbox_url = _get_latest_pdfbox_url()
			sha512_url = pdfbox_url + '.sha512'
			r = urllib.request.urlopen(pdfbox_url)
			try:
				data = r.read()
			except:
				raise RuntimeError('error retrieving %s' % pdfbox_url)
			else:
				if not os.path.exists(cache_dir.as_posix()):
					cache_dir.mkdir(parents=True)
				pdfbox_path = cache_dir.joinpath(pathlib.Path(pdfbox_url).name)
				with open(pdfbox_path.as_posix(), 'wb') as f:
					f.write(data)

			r = urllib.request.urlopen(sha512_url)
			encoding = r.headers.get_content_charset('utf-8')
			try:
				sha512 = r.read().decode(encoding).strip()
			except:
				raise RuntimeError('error retrieving sha512sum')
			else:
				if not _verify_sha512(data, sha512):
					raise RuntimeError('failed to verify sha512sum')

			return pdfbox_path

	pdfbox_path = _get_pdfbox_path()
	jpype.addClassPath(pdfbox_path)
	if not jpype.isJVMStarted():
		try:
			jpype.startJVM(convertStrings=False)
		except TypeError as ex:
			print("TypeError raised: {}.".format(ex))
		except OSError as ex:
			print("OSError raised: {}.".format(ex))

def finalize_java_vm():
	if jpype.isJVMStarted():
		jpype.shutdownJVM()

def basic_operation():
	import org.apache.pdfbox
	import java.io

	pdf_filepath = "./DeepLearning.pdf"
	page_no = 0

	#--------------------
	# Document.
	try:
		document = org.apache.pdfbox.pdmodel.PDDocument.load(java.io.File(pdf_filepath))
	except java.io.FileNotFoundException as ex:
		print("File not found, {}: {}.".format(pdf_filepath, ex))
		return

	print("#pages = {}.".format(document.getNumberOfPages()))

	print("Is encrypted = {}.".format(document.isEncrypted()))
	print("Is all security to be removed = {}.".format(document.isAllSecurityToBeRemoved()))
	print("Document ID = {}.".format(document.getDocumentId()))
	print("Version = {}.".format(document.getVersion()))
	print("Document information = {}.".format(document.getDocumentInformation()))
	print("Encryption = {}.".format(document.getEncryption()))
	print("Current access permission = {}.".format(document.getCurrentAccessPermission()))

	#--------------------
	# Page.
	try:
		page = document.getPage(page_no)
	except java.lang.IndexOutOfBoundsException as ex:
		print("Page {} not found in {}: {}.".format(page_no, pdf_filepath, ex))
		return

	print("Metadata = {}.".format(page.getMetadata()))
	print("Class = {}.".format(page.getClass()))
	#print("COSObject = {}.".format(page.getCOSObject()))
	#print("Contents = {}.".format(page.getContents()))
	#print("Content streams = {}.".format(page.getContentStreams()))
	#print("Annotations = {}.".format(page.getAnnotations()))
	#print("Actions = {}.".format(page.getActions()))
	#print("Struct parents = {}.".format(page.getStructParents()))
	#print("Resources = {}.".format(page.getResources()))
	#print("ResourceCache = {}.".format(page.getResourceCache()))
	print("User unit = {}.".format(page.getUserUnit()))
	print("Matrix = {}.".format(page.getMatrix()))
	print("Rotation = {}.".format(page.getRotation()))
	print("MediaBox = {}.".format(page.getMediaBox()))
	print("CropBox = {}.".format(page.getCropBox()))
	print("TrimBox = {}.".format(page.getTrimBox()))
	print("Bounding box = {}.".format(page.getBBox()))
	print("Art box = {}.".format(page.getArtBox()))
	print("Bleed box = {}.".format(page.getBleedBox()))

	#--------------------
	# Page image.
	dpi = 300

	pdfRenderer = org.apache.pdfbox.rendering.PDFRenderer(document)
	bim = pdfRenderer.renderImageWithDPI(page_no, dpi, org.apache.pdfbox.rendering.ImageType.RGB)

	# Save the page to an image file.
	if False:
		org.apache.pdfbox.tools.imageio.ImageIOUtil.writeImage(bim, "{}-{}.png".format(os.path.splitext(pdf_filepath)[0], page_no), dpi)

	# Retrieve data as numpy array of RGB values packed into int32.
	image_height, image_width = bim.getHeight(), bim.getWidth()
	image_data = bim.getRGB(0, 0, image_width, image_height, None, 0, image_width)[:]
	page_image = np.frombuffer(memoryview(image_data), np.uint8).reshape(image_height, image_width, 4)[..., :3]  # HxWxC.

	if False:
		import matplotlib.pyplot as plt
		plt.figure(figsize=(8, 10), tight_layout=True)
		plt.imshow(page_image)
		plt.axis('off')
		plt.show()

	#--------------------
	document.close()

def extract_paragraph_example():
	import org.apache.pdfbox
	import java.io

	pdf_filepath = "/path/to/sample.pdf"
	page_no = 0

	#--------------------
	try:
		document = org.apache.pdfbox.pdmodel.PDDocument.load(java.io.File(pdf_filepath))
	except java.io.FileNotFoundException as ex:
		print("File not found, {}: {}.".format(pdf_filepath, ex))
		return

	try:
		page = document.getPage(page_no)
	except java.lang.IndexOutOfBoundsException as ex:
		print("Page {} not found in {}: {}.".format(page_no, pdf_filepath, ex))
		return
	print("Bounding box = {}.".format(page.getBBox()))

	textStripper = org.apache.pdfbox.text.PDFTextStripper()
	textStripper.setParagraphStart("/t")
	textStripper.setSortByPosition(True)

	for line in textStripper.getText(document).split(textStripper.getParagraphStart()):
		print("-------------------------------------------")
		print(line)

	#print("Paragraph:\n{}.".format(paragraphs))

	#--------------------
	document.close()

def extract_text_in_region_example():
	import org.apache.pdfbox
	import java.io
	import java.awt.geom

	pdf_filepath = "/path/to/sample.pdf"
	page_no = 0
	targetRect = [100, 100, 400, 400]  # (x, y, width, height).

	#--------------------
	try:
		document = org.apache.pdfbox.pdmodel.PDDocument.load(java.io.File(pdf_filepath))
	except java.io.FileNotFoundException as ex:
		print("File not found, {}: {}.".format(pdf_filepath, ex))
		return

	try:
		page = document.getPage(page_no)
	except java.lang.IndexOutOfBoundsException as ex:
		print("Page {} not found in {}: {}.".format(page_no, pdf_filepath, ex))
		return
	print("Bounding box = {}.".format(page.getBBox()))

	# Only the text completely enclosed in a region are extracted.
	rect = java.awt.geom.Rectangle2D.Float(*targetRect)
	textStripper = org.apache.pdfbox.text.PDFTextStripperByArea()
	textStripper.addRegion("region", rect)
	textStripper.extractRegions(page)
	extractedText = textStripper.getTextForRegion("region")

	print("Text extracted from a region {}:\n{}.".format(targetRect, extractedText))

	#--------------------
	document.close()

def main():
	# The coordinate system:
	#	Origin: top-left, +X-axis: rightward, +Y-axis: downward.

	# Initialize JAVA VM.
	initialize_java_vm()

	#--------------------
	basic_operation()

	#extract_paragraph_example()  # Not correctly working.
	#extract_text_in_region_example()

	#--------------------
	# finalize JAVA VM.
	finalize_java_vm()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
