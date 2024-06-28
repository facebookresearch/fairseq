import requests
import json
import io
import base64

import numpy as np

from pose_format import Pose


headers = {
  'Content-Type': 'application/json'
}

# pose_path = '/home/zifjia/test_full.pose'
# with open(pose_path, "rb") as f:
#   buffer = f.read()
#   pose = Pose.read(buffer)

# memory_file = io.BytesIO()
# pose.write(memory_file)
# encoded = base64.b64encode(memory_file.getvalue())
# pose_data = encoded.decode('ascii')

# payload = json.dumps({
#   "pose": pose_data,
# })

payload = json.dumps({
  "text": "In Baar gibt es eine Höhle.",
})

url = "http://172.23.63.59:3030/api/embed/text"

response = requests.request("GET", url, headers=headers, data=payload)

print(response)

data = json.loads(response.text)

embeddings = np.asarray(data['embeddings'])
print(embeddings)
print(embeddings.shape)
