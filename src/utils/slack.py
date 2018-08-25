import os
import json
import urllib
from urllib import request

WEBHOOK_URL=os.getenv('SLACK_WEBHOOK_URL')

def notify(message):

    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        'channel': '#training-result',
        'text': json.dumps(message),
    }
    req = request.Request(WEBHOOK_URL, data=json.dumps(payload).encode('utf-8'), method='POST', headers=headers)
    try:
        request.urlopen(req)
    except urllib.error.HTTPError as e:
        print(e.code)
        print(e.read())