import requests
import json
# import jsonpickle
import base64

from pose_format import Pose


# url = "http://127.0.0.1:5000/api/text2pose"
url = "https://pub.cl.uzh.ch/demo/text2pose/"

payload = json.dumps({
  "text": "In Baar gibt es eine Höhle.",
  "language_code": "dsgs"
})
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

data = json.loads(response.text)

print(data['language_code'])
print(data['text'])

# pose = jsonpickle.decode(data['pose'])
decoded = base64.b64decode(data['pose'])
pose = Pose.read(decoded)

output_path = '/home/zifjia/test.pose'
print(f'Writing {pose} of shape {pose.body.data.shape} to {output_path} ...')
with open(output_path, "wb") as f:
    pose.write(f)