import requests
import json
import io
import base64

import numpy as np

from pose_format import Pose


headers = {
  'Content-Type': 'application/json'
}

pose_path = '/home/zifjia/test_full.pose'
with open(pose_path, "rb") as f:
  buffer = f.read()
  pose = Pose.read(buffer)

memory_file = io.BytesIO()
pose.write(memory_file)
encoded = base64.b64encode(memory_file.getvalue())
pose_data = encoded.decode('ascii')

# url = "http://127.0.0.1:8081/api/embed/pose"
# url = "http://172.23.144.112:9095/api/embed/pose"
# payload = json.dumps({
#   "pose": pose_data,
# })

url = "http://172.23.144.112:9095/api/embed/text"
payload = json.dumps({
  "text": "In Baar gibt es eine Höhle.",
})

response = requests.request("GET", url, headers=headers, data=payload)

print(response)

data = json.loads(response.text)

embedding = np.asarray(data['embedding'])
print(embedding)
print(embedding.shape)
