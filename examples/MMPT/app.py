from flask import Flask, request
from flask_cors import CORS

import io
import base64

from demo_sign import embed_pose, embed_text


app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.route('/api/embed/<modality>', methods=['GET'])
def translate(modality):
    payload = request.get_json()

    if modality == 'text':
        text = payload.get('text', '')
        embedding = embed_text(text)

    return {
        'embedding': embedding,
    }

if __name__ == '__main__':
    port = int(environ.get('PORT', 3033))
    with app.app_context():
        app.run(threaded=False,
                debug=False,
                port=port)
