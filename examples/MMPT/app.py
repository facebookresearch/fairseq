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
