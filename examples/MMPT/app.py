from flask import Flask, request
from flask_cors import CORS

import torch
from sockeye import inference, model
from sockeye.output_handler import PoseOutputHandler

import io
import base64
# import jsonpickle


app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

model_path = './models/signsuisse'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

models, source_vocabs, target_vocabs, pose_data_cfg_fname = model.load_models(
    device=device,
    model_folders=[model_path],
    checkpoints=[23], # best
    inference_only=True
)

for model in models:
    model.eval()

translator = inference.Translator(device=device,
                                  ensemble_mode='linear',
                                  scorer=inference.CandidateScorer(),
                                  output_scores=True,
                                  batch_size=1,
                                  beam_size=5,
                                  beam_search_stop='all',
                                  nbest_size=1,
                                  models=models,
                                  source_vocabs=source_vocabs,
                                  target_vocabs=target_vocabs,
                                  pose_data_cfg_fname=pose_data_cfg_fname)

@app.route('/api/text2pose/', methods=['POST'])
def translate():
    payload = request.get_json()
    language_code = payload.get('language_code', 'dsgs') # dsgs, lsf-ch, lis-ch
    text = payload.get('text', 'In Baar gibt es eine Höhle.')

    text_input = f'<{language_code}> {text}'
    input = inference.make_input_from_plain_string(0, text_input)
    output = translator.translate([input])[0]

    # output_dir = '/shares/volk.cl.uzh/zifjia/easier-continuous-translation/'
    output_dir = None
    output_handler = PoseOutputHandler(output_dir)
    pose = output_handler.handle(input, output)

    memory_file = io.BytesIO()
    pose.write(memory_file)
    # https://stackoverflow.com/questions/40000495/how-to-encode-bytes-in-json-json-dumps-throwing-a-typeerror
    encoded = base64.b64encode(memory_file.getvalue())
    pose_data = encoded.decode('ascii')

    return {
        'language_code': language_code,
        'text': text,
        # 'pose': jsonpickle.encode(pose),
        'pose': pose_data,
    }

if __name__ == '__main__':
    port = int(environ.get('PORT', 3031))
    with app.app_context():
        app.run(threaded=False,
                debug=False,
                port=port)
