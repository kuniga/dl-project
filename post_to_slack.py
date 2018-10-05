import json
import os
import urllib.request

webhook_url = os.getenv("WEBHOOK_URL")
user_id = os.getenv("SLACK_USER_ID")

obj = {
    "channel": "#test",
    "username": "bot",
    "text": "Training complete. <@{}>".format(user_id),
}

data = bytes("payload={}".format(json.dumps(obj)), "utf-8")

req = urllib.request.Request(url=webhook_url, data=data, method="POST")

with urllib.request.urlopen(req) as f:
    print(f.read().decode("utf-8"))
