#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk

# REF [site] >> https://zetcode.com/python/gtk/
def simple_example_1():
	win = Gtk.Window()
	win.connect("destroy", Gtk.main_quit)
	win.show()
	Gtk.main()

# REF [site] >> https://zetcode.com/python/gtk/
def simple_example_2():
	class MyWindow(Gtk.Window):
		def __init__(self):
			super(MyWindow, self).__init__()

			self.init_ui()

		def init_ui(self):
			self.set_icon_from_file("web.png")

			self.set_title("Icon")
			self.set_default_size(280, 180)
			
			self.connect("destroy", Gtk.main_quit)

	win = MyWindow()
	win.show()
	Gtk.main()

def main():
	simple_example_1()
	simple_example_2()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
