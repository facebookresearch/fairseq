import requests
import json
# import jsonpickle
import base64

from pose_format import Pose


url = "http://127.0.0.1:5000/api/text2pose"
# url = "http://172.16.0.71:5000/api/embed/text"

payload = json.dumps({
  "text": "In Baar gibt es eine Höhle.",
})
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("GET", url, headers=headers, data=payload)

print(response)

data = json.loads(response.text)

print(data)

# decoded = base64.b64decode(data['pose'])
# pose = Pose.read(decoded)

# output_path = '/home/zifjia/test.pose'
# print(f'Writing {pose} of shape {pose.body.data.shape} to {output_path} ...')
# with open(output_path, "wb") as f:
#     pose.write(f)