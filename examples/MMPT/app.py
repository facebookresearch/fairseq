import io
import base64

from flask import Flask, request
from flask_cors import CORS

from pose_format import Pose

from demo_sign import embed_pose, embed_text


app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.route('/api/embed/<modality>', methods=['GET'])
def embed(modality):
    '''
    Returns a list of embedding(s) of the JSON data of given modality.

    URL Parameters:
        modality (string): "pose" or "text".

    JSON keys:
        model_name: the name of the model checkpoint to use, "default" or "asl_citizen".
        pose: a list of pose data, i.e., Pose object encoded by the base64 library, see an example at
            https://colab.research.google.com/drive/1r8GtyZOJoy_tSu62tvi7Zi2ogxcqlcsz#scrollTo=kbqu1vgUuevx.
        text: a list of text strings.

    Returns:
        embeddings (list): a list of embedding(s).
    '''
    payload = request.get_json()
    model_name = payload.get('model_name', 'default')

    if modality == 'pose':
        pose_data = payload.get('pose')
        pose_data = pose_data if type(pose_data) == list else [pose_data]
        poses = [Pose.read(base64.b64decode(pose)) for pose in pose_data]
        embedding = embed_pose(poses, model_name=model_name)
    elif modality == 'text':
        text = payload.get('text')
        embedding = embed_text(text, model_name=model_name)

    return {
        'embeddings': embedding.tolist(),
    }

if __name__ == '__main__':
    port = int(environ.get('PORT', 3033))
    with app.app_context():
        app.run(threaded=False,
                debug=False,
                port=port)
