from flask import Flask
from flask import request, send_file, render_template

app = Flask(__name__)

@app.route('/')
def hello():
    # return os.getcwd()
    return 'Hello MotorSense!'

@app.route('/list')
def list():
    """
    sensor_path = "accel_00099999"
    files = os.listdir(sensor_path)
    sample_files = [f for f in files if os.path.isfile(os.path.join(sensor_path, f))]
    sample_files.sort()

    response = ""
    for x in sample_files:
        response += ('<a href="fft/{0}">{0}</a><br/>'.format(x))
        #response += ('<a href="{0}fft/{1}">{1}</a><br/>'.format(request.url_root, x))
    return response
    """

    sensor_path = '.'
    files = os.listdir(sensor_path)
    sample_files = [f for f in files if os.path.join(sensor_path, f)]
    sample_files.sort()

    response = ""
    for file in sample_files:
    	if os.path.isfile(file):
            response += ('<a href="fft/{0}">{0}</a><br/>'.format(file))
            #response += ('<a href="{0}fft/{1}">{1}</a><br/>'.format(request.url_root, file))
    	else:
            response += ('<a href="list/{0}">{0}</a><br/>'.format(file))
    return response

def save_accel_fft(sensor_filepath, fft_image_filepath):
	# 800 Hz, 10 secs, 3 channels, 2 bytes = 48000 bytes.
	Fs = 800  # Hz.
	timespan = 10  # secs.
	num_channels = 3  # channels.

	divisor = 256.0  # ADXL345.
	#divisor = 3276.7  # ADXL357.

	signals = sensor_util.load_accel_file(sensor_filepath, Fs, timespan, num_channels)
	signals = pd.DataFrame(signals) / divisor

	num_samples = signals.shape[0]
	"""
	#signal = signals.x  # X axis.
	signal = signals.y  # Y axis.
	#signal = signals.z  # Z axis.
	sensor_util.save_fft(fft_image_filepath, signal, Fs, num_samples)
	"""
	sensor_util.save_fft_3ch(fft_image_filepath, signals, Fs, num_samples)

@app.route('/fft/<path:filename>', methods=['GET'])
def show_fft(filename):
	if 'GET' == request.method:
		print('***********************FFT', request.path, request.script_root, request.url, request.base_url, request.url_root)
		print('***********************FFT', filename)

		fft_image_filepath = './accel_fft.png'
		sensor_filepath = SENOR_BASE_DIR + '/' + filename

		save_accel_fft(sensor_filepath, fft_image_filepath)
		return send_file(fft_image_filepath)

@app.route('/upload/<int:sensor_id>', methods=['GET', 'POST'])
def upload_file(sensor_id):
    if request.method == 'POST':
        timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        dir_path = './accel_{0:08}'.format(sensor_id)
        make_dir(dir_path)
        filepath = './accel_{0:08}/accel_{0:08}_{1}.raw'.format(sensor_id, timestamp)

        f = request.files['sensor_data']
        f.save(filepath)
        return "OKAY"

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

@app.route('/list_all', defaults={'req_path': ''})
@app.route('/list_all/<path:req_path>')
def dirtree(req_path):
	#path = os.path.expanduser(u'~')
	path = os.path.abspath(req_path)
	return render_template('dirtree.html', tree=make_tree(path))

@app.route('/', defaults={'dir_path': ''})
@app.route('/<path:dir_path>')
def list_dir(dir_path):
	print('*****1', request.path, request.script_root, request.url, request.base_url, request.url_root)
	print('*****2', dir_path)

	if bool(dir_path.strip()):
		local_dir_path = SENOR_BASE_DIR + '/' + dir_path
	else:
		local_dir_path = SENOR_BASE_DIR
	print('*****3', local_dir_path)
	fd_list = os.listdir(local_dir_path)
	fd_list.sort()

	response = ''
	for fd in fd_list:
		fd_path = os.path.join(local_dir_path, fd)
		if os.path.isfile(fd_path):
			if fd.endswith('.raw'):
				response += ('<a href="{0}fft/{1}{2}">{2}</a><br/>'.format(request.url_root, dir_path, fd))
			else:
				response += ('{}<br/>'.format(fd))
		else:
			response += ('<a href="{0}/">{0}</a><br/>'.format(fd))
	return response

if __name__ == "__main__":
    app.run("0.0.0.0", 6788)
