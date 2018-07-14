#!/usr/bin/env python

# REF [site] >>
#	http://flask.pocoo.org/docs/
#	https://www.tutorialspoint.com/flask/index.htm

from flask import Flask
from flask import request, send_file, send_from_directory, render_template
import os

app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER'] = './'
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.route('/')
def hello():
    return 'Hello Flask!'

@app.route('/python/<path:filepath>', methods=['GET'])
def show_python(filepath):
	return send_file(filepath)

@app.route('/list/', defaults={'dir_path': ''})
@app.route('/list/<path:dir_path>')
def list_dir(dir_path):
	#print('*****1', request.path, request.script_root, request.url, request.base_url, request.url_root)
	#print('*****2', dir_path)

	if bool(dir_path.strip()):
		local_dir_path = dir_path
	else:
		local_dir_path = '.'
	fd_list = os.listdir(local_dir_path)
	fd_list.sort()

	response = ''
	for fd in fd_list:
		fd_path = os.path.join(local_dir_path, fd)
		if os.path.isfile(fd_path):
			if fd.endswith('.py'):
				url = os.path.join(request.url_root, 'python', dir_path, fd)
				response += ('<a href="{0}">{1}</a><br/>'.format(url, fd))
			else:
				response += ('{}<br/>'.format(fd))
		else:
			url = os.path.join(request.url, fd)
			response += ('<a href="{0}/">{1}</a><br/>'.format(url, fd))
	return response

def make_tree(path):
	tree = dict(name=os.path.basename(path), children=[])
	try:
		lst = os.listdir(path)
	except OSError:
		pass  # Ignore errors.
	else:
		for name in lst:
			fn = os.path.join(path, name)
			if os.path.isdir(fn):
				tree['children'].append(make_tree(fn))
			else:
				tree['children'].append(dict(name=name))
	return tree

@app.route('/list_all/', defaults={'dir_path': ''})
@app.route('/list_all/<path:dir_path>')
def list_all_dirs(dir_path):
	#path = os.path.expanduser(u'~')
	path = os.path.abspath(dir_path)
	return render_template('dirtree.html', tree=make_tree(path))

@app.route('/download/<path:filename>', methods=['GET', 'POST'])
def download_file(filename):
	return send_from_directory(directory='.', filename=filename, as_attachment=True)
	#return send_file(path, as_attachment=True)

# REF [site] >> http://flask.pocoo.org/docs/1.0/patterns/fileuploads/
@app.route('/upload/', methods=['GET', 'POST'])
def upload_file():
	if 'POST' == request.method:
		#file = request.files['file']
		file = request.files.get('file', None)
		if file is not None:
			filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename + '.uploaded')
			file.save(filepath)
		else:
			print('File not found: {}'.format('file'))

		#file = request.files['conf_file']
		file = request.files.get('conf_file', None)
		if file is not None:
			filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename + '.uploaded')
			file.save(filepath)
		else:
			print('File not found: {}'.format('conf_file'))

		#file = request.files['json_file']
		file = request.files.get('json_file', None)
		if file is not None:
			filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename + '.uploaded')
			file.save(filepath)
		else:
			print('File not found: {}'.format('json_file'))

		return 'OK', 201

#%%------------------------------------------------------------------

# Usage:
#	python simple_flask_server.py
#
#	flask run --help
#	python -m flask run --help
#
#	FLASK_APP=simple_flask_server.py flask run
#	FLASK_APP=simple_flask_server.py flask run --host=0.0.0.0 --port=8888
#
#	export FLASK_APP=simple_flask_server.py
#	set FLASK_APP=simple_flask_server.py
#	flask run
#	flask run --host=0.0.0.0 --port=8888
#
#	export FLASK_APP=simple_flask_server.py
#	set FLASK_APP=simple_flask_server.py
#	python -m flask run
#	python -m flask run --host=0.0.0.0 --port=8888

if '__main__' == __name__:
	try:
		app.run(host='0.0.0.0', port=8888, debug=False)
	except:
		#ex = sys.exc_info()  # (type, exception object, traceback).
		##print('{} raised: {}.'.format(ex[0], ex[1]))
		#print('{} raised: {}.'.format(ex[0].__name__, ex[1]))
		#traceback.print_tb(ex[2], limit=None, file=sys.stdout)
		#traceback.print_exception(*sys.exc_info(), limit=None, file=sys.stdout)
		traceback.print_exc(limit=None, file=sys.stdout)
