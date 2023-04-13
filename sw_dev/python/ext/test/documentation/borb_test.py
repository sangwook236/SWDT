#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import typing
from pathlib import Path
from decimal import Decimal
from borb.io.read.types import Name, String, Dictionary
from borb.pdf.canvas.geometry.rectangle import Rectangle
from borb.pdf import ConnectedShape
from borb.pdf import Alignment
from borb.pdf import FixedColumnWidthTable
from borb.pdf import FlexibleColumnWidthTable
from borb.pdf import LineArtFactory
from borb.pdf import UnorderedList
from borb.pdf import TextField
from borb.pdf import CheckBox
from borb.pdf import CountryDropDownList
from borb.pdf import JavaScriptPushButton
from borb.pdf import SingleColumnLayout
from borb.pdf import PageLayout
from borb.pdf import Paragraph
from borb.pdf import Document
from borb.pdf import Page
from borb.pdf import PDF
from borb.pdf import HexColor, RGBColor
from borb.pdf import Image
from borb.toolkit import SimpleTextExtraction
from borb.toolkit import RegularExpressionTextExtraction
from borb.toolkit import LocationFilter
from borb.toolkit import PDFMatch
from borb.toolkit import TFIDFKeywordExtraction
from borb.toolkit import TextRankKeywordExtraction
from borb.toolkit import ENGLISH_STOP_WORDS
from borb.toolkit import ColorExtraction
from borb.toolkit import FontExtraction

# REF [site] >> https://github.com/jorisschellekens/borb
def hello_world_example():
	# Create an empty Document.
	pdf = Document()

	# Add an empty Page.
	page = Page()
	pdf.add_page(page)

	# Use a PageLayout (SingleColumnLayout in this case).
	layout = SingleColumnLayout(page)

	# Add a Paragraph object.
	layout.add(Paragraph("Hello World!"))
		
	# Store the PDF.
	with open(Path("./output.pdf"), "wb") as fd:
		PDF.dumps(fd, pdf)

# REF [site] >> https://github.com/jorisschellekens/borb-examples#43-adding-formfield-objects-to-a-pdf
def adding_formfield_objects_to_pdf_example():
	# Create Document.
	doc: Document = Document()

	# Create Page.
	page: Page = Page()

	# Add Page to Document.
	doc.add_page(page)

	# Set a PageLayout.
	layout: PageLayout = SingleColumnLayout(page)

	# Add FixedColumnWidthTable containing Paragraph and TextField objects.
	layout.add(
		FixedColumnWidthTable(number_of_columns=2, number_of_rows=5)
			.add(Paragraph("Name:"))
			#.add(TextField(field_name="name"))
			.add(TextField(field_name="name", font_color=HexColor("f1cd2e")))
			.add(Paragraph("Firstname:"))
			#.add(TextField(field_name="firstname"))
			.add(TextField(field_name="firstname", font_color=HexColor("f1cd2e")))
			.add(Paragraph("Country"))
			#.add(TextField(field_name="country"))
			#.add(TextField(field_name="country", value="Belgium"))  # Add TextField already pre-filled with 'Belgium'.
			#.add(DropDownList(field_name="country", possible_values=["Belgium", "Canada", "Denmark", "Estonia"],))
			.add(CountryDropDownList(field_name="country"))
			.add(Paragraph("Do you want to receive promotional emails?"))
			.add(CheckBox())
			.add(Paragraph(" "))
			.add(
				JavaScriptPushButton(
					text="Popup!",
					javascript="app.alert('Hello World!', 3)",
					horizontal_alignment=Alignment.RIGHT,
				)
			)
			.set_padding_on_all_cells(Decimal(2), Decimal(2), Decimal(2), Decimal(2))
			.no_borders()
	)

	# Store.
	with open("./output.pdf", "wb") as fd:
		PDF.dumps(fd, doc)

# REF [site] >> https://github.com/jorisschellekens/borb-examples#5-working-with-existing-pdfs
def working_with_existing_pdfs_example():
	# Create Document.
	doc: Document = Document()

	# Create Page.
	page: Page = Page()

	# Add Page to Document.
	doc.add_page(page)

	# Set a PageLayout.
	layout: PageLayout = SingleColumnLayout(page)

	# Add a Paragraph.
	layout.add(Paragraph(
"""
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. 
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. 
Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
"""
	))

	# Set the /Info dictionary.
	doc["XRef"]["Trailer"][Name("Info")] = Dictionary()

	# Set the /Author.
	doc["XRef"]["Trailer"]["Info"][Name("Author")] = String("Joris Schellekens")

	with open("./output.pdf", "wb") as fd:
		PDF.dumps(fd, doc)

	#--------------------
	# Extracting meta-information from a PDF.

	# Read the Document.
	doc: typing.Optional[Document] = None
	with open("./output.pdf", "rb") as fd:
		doc = PDF.loads(fd)

	# Check whether we have read a Document.
	assert doc is not None

	# Print the \Author key from the \Info dictionary.
	print(f"Author: {doc.get_document_info().get_author()}.")
	# Print the \Producer key from the \Info dictionary.
	print(f"Producer: {doc.get_document_info().get_producer()}.")
	# Print the ID using XMP meta info.
	print(f"ID: {doc.get_xmp_document_info().get_document_id()}.")

	#--------------------
	# Extracting text from a PDF.

	# Read the Document.
	doc: typing.Optional[Document] = None
	l: SimpleTextExtraction = SimpleTextExtraction()
	with open("./output.pdf", "rb") as fd:
		doc = PDF.loads(fd, [l])

	# Check whether we have read a Document.
	assert doc is not None

	# Print the text on the first Page.
	print("First page:")
	print(l.get_text()[0])

	#--------------------
	# Extracting text using regular expressions.

	# Read the Document.
	# fmt: off
	doc: typing.Optional[Document] = None
	l: RegularExpressionTextExtraction = RegularExpressionTextExtraction("[lL]orem .* [dD]olor")
	with open("./output.pdf", "rb") as fd:
		doc = PDF.loads(fd, [l])
	# fmt: on

	# Check whether we have read a Document.
	assert doc is not None

	# Print matching groups.
	for i, m in enumerate(l.get_matches()[0]):
		print(f"{i} {m.group(0)}.")
		for r in m.get_bounding_boxes():
			print(f"\tx = {r.get_x()}, y = {r.get_y()}, w = {r.get_width()}, h = {r.get_height()}.")

	#--------------------
	# Extracting text using its bounding box.

	# Define the Rectangle of interest.
	r: Rectangle = Rectangle(Decimal(59), Decimal(740), Decimal(99), Decimal(11))

	# Define SimpleTextExtraction.
	l0: SimpleTextExtraction = SimpleTextExtraction()

	# Apply a LocationFilter on top of SimpleTextExtraction.
	l1: LocationFilter = LocationFilter(r)
	l1.add_listener(l0)

	# Read the Document.
	doc: typing.Optional[Document] = None
	with open("./output.pdf", "rb") as fd:
		doc = PDF.loads(fd, [l1])

	# Check whether we have read a Document.
	assert doc is not None

	# Print the text inside the Rectangle of interest.
	print("Text inside the Rectangle of interest:")
	print(l0.get_text()[0])

	#--------------------
	# Combining regular expressions and bounding boxes.

	# Set up RegularExpressionTextExtraction.
	# fmt: off
	l0: RegularExpressionTextExtraction = RegularExpressionTextExtraction("[nN]isi .* aliquip")
	# fmt: on

	# Process Document.
	doc: typing.Optional[Document] = None
	with open("./output.pdf", "rb") as fd:
		doc = PDF.loads(fd, [l0])
	assert doc is not None

	# Find match.
	m: typing.Optional[PDFMatch] = next(iter(l0.get_matches()[0]), None)
	assert m is not None

	# Get page width.
	w: Decimal = doc.get_page(0).get_page_info().get_width()

	# Change rectangle to get more text.
	r0: Rectangle = m.get_bounding_boxes()[0]
	r1: Rectangle = Rectangle(r0.get_x() + r0.get_width(), r0.get_y(), w - r0.get_x(), r0.get_height())

	# Process document (again) filtering by rectangle.
	l1: LocationFilter = LocationFilter(r1)
	l2: SimpleTextExtraction = SimpleTextExtraction()
	l1.add_listener(l2)
	doc: typing.Optional[Document] = None
	with open("./output.pdf", "rb") as fd:
		doc = PDF.loads(fd, [l1])
	assert doc is not None

	# Get text.
	print(l2.get_text()[0])

# REF [site] >> https://github.com/jorisschellekens/borb-examples#5-working-with-existing-pdfs
def extracting_keywords_from_existing_pdfs_example():
	# Create Document.
	doc: Document = Document()

	# Create Page.
	page: Page = Page()

	# Add Page to Document.
	doc.add_page(page)

	# Set a PageLayout.
	layout: PageLayout = SingleColumnLayout(page)

	# Add first Paragraph.
	layout.add(Paragraph("What is Lorem Ipsum?", font="Helvetica-bold"))
	layout.add(Paragraph("""
Lorem Ipsum is simply dummy text of the printing and typesetting industry. 
Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, 
when an unknown printer took a galley of type and scrambled it to make a type specimen book. 
It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. 
It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, 
and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.
"""
	))

	# Add second Paragraph.
	layout.add(Paragraph("Where does it come from?", font="Helvetica-bold"))
	layout.add(Paragraph(
"""
Contrary to popular belief, Lorem Ipsum is not simply random text. 
It has roots in a piece of classical Latin literature from 45 BC, making it over 2000 years old. 
Richard McClintock, a Latin professor at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words, 
consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, 
discovered the undoubtable source. Lorem Ipsum comes from sections 1.10.32 and 1.10.33 of "de Finibus Bonorum et Malorum" 
(The Extremes of Good and Evil) by Cicero, written in 45 BC. 
This book is a treatise on the theory of ethics, very popular during the Renaissance. 
The first line of Lorem Ipsum, "Lorem ipsum dolor sit amet..", comes from a line in section 1.10.32.
"""
	))

	# Store.
	with open("./output.pdf", "wb") as fd:
		PDF.dumps(fd, doc)

	#--------------------
	# Extracting keywords from a PDF using TF-IDF.

	l: TFIDFKeywordExtraction = TFIDFKeywordExtraction(stopwords=ENGLISH_STOP_WORDS)

	# Load.
	doc: typing.Optional[Document] = None
	with open("./output.pdf", "rb") as fd:
		doc = PDF.loads(fd, [l])

	# Check whether we have read a Document.
	assert doc is not None

	print("Keywords extracted by TF-IDF:")
	print(l.get_keywords()[0])

	#--------------------
	# Extracting keywords from a PDF using TextRank.

	l: TextRankKeywordExtraction = TextRankKeywordExtraction()

	# Load.
	doc: typing.Optional[Document] = None
	with open("./output.pdf", "rb") as fd:
		doc = PDF.loads(fd, [l])

	# Check whether we have read a Document.
	assert doc is not None

	print("Keywords extracted by TextRank:")
	print(l.get_keywords()[0])

# REF [site] >> https://github.com/jorisschellekens/borb-examples#5-working-with-existing-pdfs
def extracting_color_information_from_existing_pdfs_example():
	# Create Document.
	doc: Document = Document()

	# Create Page.
	page: Page = Page()

	# Add Page to Document.
	doc.add_page(page)

	# Set a PageLayout.
	layout: PageLayout = SingleColumnLayout(page)

	# The following code adds 3 paragraphs, each in a different color.
	layout.add(Paragraph("Hello World!", font_color=HexColor("FF0000")))
	layout.add(Paragraph("Hello World!", font_color=HexColor("00FF00")))
	layout.add(Paragraph("Hello World!", font_color=HexColor("0000FF")))

	# The following code adds 1 image.
	layout.add(
		Image(
			"https://images.unsplash.com/photo-1589606663923-283bbd309229?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8",
			width=Decimal(256),
			height=Decimal(256),
		)
	)

	# Store.
	with open("./output.pdf", "wb") as fd:
		PDF.dumps(fd, doc)

	#--------------------
	doc: typing.Optional[Document] = None
	l: ColorExtraction = ColorExtraction()
	with open("./output.pdf", "rb") as fd:
		doc = PDF.loads(fd, [l])

	# Extract colors.
	colors: typing.Dict[Color, Decimal] = l.get_color()[0]

	# Create output Document.
	doc_out: Document = Document()

	# Add Page.
	p: Page = Page()
	doc_out.add_page(p)

	# Add PageLayout.
	l: PageLayout = SingleColumnLayout(p)

	# Add Paragraph.
	l.add(Paragraph("These are the colors used in the input PDF:"))

	# Add Table.
	t: FlexibleColumnWidthTable = FlexibleColumnWidthTable(number_of_rows=3, number_of_columns=3, horizontal_alignment=Alignment.CENTERED)
	for c in colors.keys():
		t.add(
			ConnectedShape(
				LineArtFactory.droplet(Rectangle(Decimal(0), Decimal(0), Decimal(32), Decimal(32))),
				stroke_color=c,
				fill_color=c,
			)
		)
	t.set_padding_on_all_cells(Decimal(5), Decimal(5), Decimal(5), Decimal(5))
	t.no_borders()
	l.add(t)

	# Store.
	with open("./output.pdf", "wb") as fd:
		PDF.dumps(fd, doc_out)

# REF [site] >> https://github.com/jorisschellekens/borb-examples#5-working-with-existing-pdfs
def extracting_font_information_from_existing_pdfs_example():
	# Create Document.
	doc: Document = Document()

	# Create Page.
	page: Page = Page()

	# Add Page to Document.
	doc.add_page(page)

	# Set a PageLayout.
	layout: PageLayout = SingleColumnLayout(page)

	# Add UnorderedList containing a (twice nested) UnorderedList.
	for font_name in ["Helvetica", "Helvetica-Bold", "Courier"]:
		layout.add(Paragraph(f"Hello World from {font_name}!", font=font_name))

	# Store.
	with open("./output.pdf", "wb") as fd:
		PDF.dumps(fd, doc)

	#--------------------
	# Read the Document.
	doc: typing.Optional[Document] = None
	l: FontExtraction = FontExtraction()
	with open("./output.pdf", "rb") as fd:
		doc = PDF.loads(fd, [l])

	# Check whether we have read a Document.
	assert doc is not None

	# Print the names of the Fonts.
	print("Font names:")
	print(l.get_font_names()[0])

# REF [site] >> https://github.com/jorisschellekens/borb-examples#71-extracting-tables-from-a-pdf
def extracting_tables_from_pdf_example():
	raise NotImplementedError

def main():
	hello_world_example()

	# 2. Creating PDF documents from scratch.
	# 3. Container LayoutElement objects.
	# 4. Forms.
	adding_formfield_objects_to_pdf_example()
	# 5. Working with existing PDFs.
	working_with_existing_pdfs_example()
	extracting_keywords_from_existing_pdfs_example()
	extracting_color_information_from_existing_pdfs_example()
	extracting_font_information_from_existing_pdfs_example()
	# 6. Adding annotations to a PDF.
	# 7. Heuristics for PDF documents.
	#extracting_tables_from_pdf_example()  # Not yet implemented.

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
