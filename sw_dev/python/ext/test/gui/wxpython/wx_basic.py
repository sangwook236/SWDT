#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import wx

class SimpleFrame(wx.Frame):
	def __init__(self, *args, **kwargs):
		# Ensure the parent's __init__() is called.
		super(SimpleFrame, self).__init__(*args, **kwargs)

		# Create a panel in the frame.
		panel = wx.Panel(self)

		# Put some text with a larger bold font on it.
		st = wx.StaticText(panel, label="Hello World!")
		font = st.GetFont()
		font.PointSize += 10
		font = font.Bold()
		st.SetFont(font)

		# Create a sizer to manage the layout of child widgets.
		sizer = wx.BoxSizer(wx.VERTICAL)
		sizer.Add(st, wx.SizerFlags().Border(wx.TOP | wx.LEFT, 25))
		panel.SetSizer(sizer)

		# Initialize UI.
		self.InitUI()

		#--------------------
		self.SetSize((450, 350))
		self.SetTitle("Simple Example")
		self.Centre()

	def InitUI(self):
		"""
		A menu bar is composed of menus, which are composed of menu items.
		This method builds a set of menus and binds handlers to be called
		when the menu item is selected.
		"""

		# Create a menu bar.
		fileMenu = wx.Menu()
		newItem = fileMenu.Append(wx.ID_NEW, "&New")
		openItem = fileMenu.Append(wx.ID_OPEN, "&Open...")
		saveAsItem = fileMenu.Append(wx.ID_SAVE, "&Save As...")
		fileMenu.AppendSeparator()

		importMenu = wx.Menu()
		importMenu.Append(wx.ID_ANY, "Import Newsfeed List...")
		importMenu.Append(wx.ID_ANY, "mport Bookmarks...")
		importMenu.Append(wx.ID_ANY, "Import Mail...")
		fileMenu.AppendMenu(wx.ID_ANY, "I&mport", importMenu)
		fileMenu.AppendSeparator()

		if True:
			# When using a stock ID we don't need to specify the menu item's label.
			exitItem = fileMenu.Append(wx.ID_EXIT)
		else:
			exitItem = wx.MenuItem(fileMenu, 1, "&Quit\tCtrl+Q")
			exitItem.SetBitmap(wx.Bitmap("./exit.png"))
			fileMenu.Append(exitItem)

		viewMenu = wx.Menu()
		self.showStatusbarItem = viewMenu.Append(wx.ID_ANY, "Show Statusbar", "Show Statusbar", kind=wx.ITEM_CHECK)
		self.showToolbarItem = viewMenu.Append(wx.ID_ANY, "Show Toolbar", "Show Toolbar", kind=wx.ITEM_CHECK)

		viewMenu.Check(self.showStatusbarItem.GetId(), True)
		viewMenu.Check(self.showToolbarItem.GetId(), True)

		messageMenu = wx.Menu()
		# The "\t..." syntax defines an accelerator key that also triggers the same event.
		helloItem = messageMenu.Append(wx.ID_ANY, "&Hello...\tCtrl-H", "Help string shown in status bar for this menu item")
		messageMenu.AppendSeparator()
		messageItem = messageMenu.Append(wx.ID_ANY, "&Message...\tCtrl-M", "Message")
		errorItem = messageMenu.Append(wx.ID_ANY, "&Error...\tCtrl-E", "Error")
		questionItem = messageMenu.Append(wx.ID_ANY, "&Question...\tCtrl-U", "Question")
		exclamationItem = messageMenu.Append(wx.ID_ANY, "&Exclamation...\tCtrl-C", "Exclamation")

		# Now a help menu for the about item.
		helpMenu = wx.Menu()
		aboutItem = helpMenu.Append(wx.ID_ABOUT)

		# Make the menu bar and add the two menus to it. The '&' defines
		# that the next letter is the "mnemonic" for the menu item. On the
		# platforms that support it those letters are underlined and can be
		# triggered from the keyboard.
		menuBar = wx.MenuBar()
		menuBar.Append(fileMenu, "&File")
		menuBar.Append(viewMenu, "&View")
		menuBar.Append(messageMenu, "&Message")
		menuBar.Append(helpMenu, "&Help")

		# Give the menu bar to the frame.
		self.SetMenuBar(menuBar)

		#--------------------
		# Create a status bar.
		self.statusbar = self.CreateStatusBar()
		self.SetStatusText("Welcome to wxPython!")
		#self.statusbar.SetStatusText("Welcome to wxPython!")

		#--------------------
		# Create a toolbar.
		self.toolbar = self.CreateToolBar()
		self.toolbar.AddTool(1, "Tool 1", wx.Bitmap("./right.png"), wx.Bitmap("./wrong.png"), kind=wx.ITEM_RADIO, shortHelp="Simple Tool 1")
		#self.toolbar.AddStretchableSpace()
		self.toolbar.AddTool(1, "Tool 2", wx.Bitmap("./right.png"), wx.Bitmap("./wrong.png"), kind=wx.ITEM_CHECK, shortHelp="Simple Tool 2")
		#self.toolbar.AddStretchableSpace()
		self.toolbar.AddTool(1, "Tool 3", wx.Bitmap("./right.png"), wx.Bitmap("./wrong.png"), kind=wx.ITEM_NORMAL, shortHelp="Simple Tool 3")
		self.toolbar.Realize()

		#--------------------
		# Finally, associate a handler function with the EVT_MENU event for each of the menu items.
		# That means that when that menu item is activated then the associated handler function will be called.
		self.Bind(wx.EVT_MENU, self.OnNew, newItem)
		self.Bind(wx.EVT_MENU, self.OnOpen, openItem)
		self.Bind(wx.EVT_MENU, self.OnSaveAs, saveAsItem)
		self.Bind(wx.EVT_MENU, self.OnExit, exitItem)
		self.Bind(wx.EVT_MENU, self.OnToggleStatusBar, self.showStatusbarItem)
		self.Bind(wx.EVT_MENU, self.OnToggleToolBar, self.showToolbarItem)
		self.Bind(wx.EVT_MENU, self.OnHello, helloItem)
		self.Bind(wx.EVT_MENU, self.OnMessage, messageItem)
		self.Bind(wx.EVT_MENU, self.OnError, errorItem)
		self.Bind(wx.EVT_MENU, self.OnQuestion, questionItem)
		self.Bind(wx.EVT_MENU, self.OnExclamation, exclamationItem)
		self.Bind(wx.EVT_MENU, self.OnAbout, aboutItem)

		self.Bind(wx.EVT_PAINT, self.OnPaint)

	def OnNew(self, event):
		wx.MessageBox("New MenuItem Clicked")

	def OnOpen(self, event):
		# REF [site] >> https://docs.wxpython.org/wx.FileDialog.html
		with wx.FileDialog(self, "Open File", wildcard="PNG files (*.png)|*.png|JPG files (*.jpg)|*.jpg|BMP and GIF files (*.bmp;*.gif)|*.bmp;*.gif|All files (*.*)|*.*", style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
			if dlg.ShowModal() == wx.ID_CANCEL:
				return

			filepath = dlg.GetPath()
			try:
				with open(filepath, "r") as fd:
					wx.MessageBox("{} opened".format(filepath))
			except IOError as ex:
				wx.LogError("Cannot open {}: {}.".filepath(filepath, ex))

	def OnSaveAs(self, event):
		# REF [site] >> https://docs.wxpython.org/wx.FileDialog.html
		with wx.FileDialog(self, "Save File", wildcard="PNG files (*.png)|*.png|JPG files (*.jpg)|*.jpg|BMP and GIF files (*.bmp;*.gif)|*.bmp;*.gif|All files (*.*)|*.*", style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
			if dlg.ShowModal() == wx.ID_CANCEL:
				return

			filepath = dlg.GetPath()
			try:
				with open(filepath, "w") as fd:
					wx.MessageBox("{} saved".format(filepath))
			except IOError as ex:
				wx.LogError("Cannot save to {}: {}.".format(filepath, ex))

	def OnExit(self, event):
		self.Close(True)

	def OnToggleStatusBar(self, event):
		if self.showStatusbarItem.IsChecked():
			self.statusbar.Show()
		else:
			self.statusbar.Hide()

	def OnToggleToolBar(self, event):
		if self.showToolbarItem.IsChecked():
			self.toolbar.Show()
		else:
			self.toolbar.Hide()

	def OnHello(self, event):
		wx.MessageBox("Hello again from wxPython")

	def OnMessage(self, event):
		dial = wx.MessageDialog(None, "Download completed", "Info", wx.OK)
		dial.ShowModal()

	def OnError(self, event):
		dlg = wx.MessageDialog(None, "Error loading file", "Error", wx.OK | wx.ICON_ERROR)
		dlg.ShowModal()

	def OnQuestion(self, event):
		dlg = wx.MessageDialog(None, "Are you sure to quit?", "Question", wx.YES_NO | wx.NO_DEFAULT | wx.ICON_QUESTION)
		dlg.ShowModal()

	def OnExclamation(self, event):
		dlg = wx.MessageDialog(None, "Unallowed operation", "Exclamation", wx.OK | wx.ICON_EXCLAMATION)
		dlg.ShowModal()

	def OnAbout(self, event):
		wx.MessageBox("This is a simple wxPython sample",
			"About Simple Example",
			wx.OK | wx.ICON_INFORMATION)

	def OnPaint(self, event):
		dc = wx.PaintDC(self)
		dc.SetPen(wx.Pen("#d4d4d4"))

		dc.SetBrush(wx.Brush("#c56c00"))
		dc.DrawRectangle(10, 15, 90, 60)

		dc.SetBrush(wx.Brush("#1ac500"))
		dc.DrawRectangle(130, 15, 90, 60)

		dc.SetBrush(wx.Brush("#539e47"))
		dc.DrawRectangle(250, 15, 90, 60)

		dc.SetBrush(wx.Brush("#004fc5"))
		dc.DrawRectangle(10, 105, 90, 60)

		dc.SetBrush(wx.Brush("#c50024"))
		dc.DrawRectangle(130, 105, 90, 60)

		dc.SetBrush(wx.Brush("#9e4757"))
		dc.DrawRectangle(250, 105, 90, 60)

		dc.SetBrush(wx.Brush("#5f3b00"))
		dc.DrawRectangle(10, 195, 90, 60)

		dc.SetBrush(wx.Brush("#4c4c4c"))
		dc.DrawRectangle(130, 195, 90, 60)

		dc.SetBrush(wx.Brush("#785f36"))
		dc.DrawRectangle(250, 195, 90, 60)

# REF [site] >>
#	https://www.wxpython.org/pages/overview/
#	https://zetcode.com/wxpython/
def simple_example():
	# When this module is run (not imported) then create the app, the frame, show it, and start the event loop.
	app = wx.App()
	frame = SimpleFrame(None, title="Simple Example !!!")
	frame.Show()
	app.MainLoop()

def main():
	simple_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
