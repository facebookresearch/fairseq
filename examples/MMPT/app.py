import io
import base64

from flask import Flask, request
from flask_cors import CORS

from pose_format import Pose

from demo_sign import embed_pose, embed_text


app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.route('/api/embed/<modality>', methods=['GET'])
def translate(modality):
    payload = request.get_json()

    if modality == 'pose':
        model = payload.get('model')
        pose_data = payload.get('pose')
        pose_data = pose_data if type(pose_data) == list else [pose_data]
        poses = [Pose.read(base64.b64decode(pose)) for pose in pose_data]
        embedding = embed_pose(poses, model=model)
    elif modality == 'text':
        text = payload.get('text')
        embedding = embed_text(text)

    return {
        'embeddings': embedding.tolist(),
    }

if __name__ == '__main__':
    port = int(environ.get('PORT', 3033))
    with app.app_context():
        app.run(threaded=False,
                debug=False,
                port=port)
