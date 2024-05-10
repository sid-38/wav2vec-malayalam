import os
from flask import Flask, flash, request, redirect, url_for, send_file
import transcribe
from werkzeug.utils import secure_filename
import threading
import time


UPLOAD_FOLDER = '/code/uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3'}
running_threads = {}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def hello():
    return 'Hello World!\n'

# @app.route('/transcribe')
# def transcribe():
#     inp_file = "./mlm_00269_00156195788.wav"
#     out_file = "output.txt"
#     transcriber.transcribe(inp_file, out_file)
#     return "Transcribed!\n"

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/transcribe', methods=['GET', 'POST'])
def trancsribe_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            time_id = int(time.time())
            inp_file = os.path.join("./", str(time_id)+filename)
            out_file = os.path.join(f"{time_id}.txt")
            file.save(inp_file)
            print("Starting transcription")
            transcriber = transcribe.Transcriber(inp_file, out_file)
            # asyncio.create_task(transcriber.transcribe())
            transcribe_thread = threading.Thread(target = transcriber.transcribe, args=())
            running_threads[time_id] = {'transcriber': transcriber, 'out_file': out_file, 'inp_file': inp_file,
                                        'thread': transcribe_thread}
            transcribe_thread.start()
            # transcriber.transcribe(os.path.join("./", filename), os.path.join("./", filename+"transcribed.txt"))
            return(f"Transcribing with {time_id}")
            # return redirect(url_for('download_file', name=filename+"transcribed.txt"))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/uploads/<name>')
def download_file(name):
    return send_file(f"./{name}.txt")

@app.route('/status/<file_id>')
def get_status(file_id):
    try:
        file_id = int(file_id)
    except ValueError:
        return "Provide the right file id"

    status = running_threads[file_id]['transcriber'].get_status()
    return f"{status[0]} out of {status[1]} with a status of {status[2]}"

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)