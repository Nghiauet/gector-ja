# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START gae_python38_app]
# [START gae_python3_app]
import unicodedata # for text normalizaton 
from difflib import ndiff # for highlight the difference

from flask import Flask, render_template, request, jsonify # flask for web interface
from model import GEC # import GEC model


# If `entrypoint` is not defined in app.yaml, App Engine will look for an app
# called `app` in `main.py`.
app = Flask(__name__)
gec = GEC(pretrained_weights_path='data/model/model_checkpoint') # init the model with pretrained weights


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/correct', methods=['POST'])
def correct(): # hander the post request retrive the text from the user request 
    text = unicodedata.normalize('NFKC', request.json['text']).replace(' ', '')
    correct_text = gec.correct(text) # correct text from model 
    diffs = list(ndiff(text, correct_text)) # compare the diff
    print(f'Correction: {text} -> {correct_text}') # print output
    return jsonify({
        'correctedText': correct_text,
        'diffs': diffs
    })


if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host='127.0.0.1', port=8080, threaded=False, use_reloader=False) # config the model in 127.0.0.1
# [END gae_python3_app]
# [END gae_python38_app]
