import os
import json
import urllib.request

key = os.environ.get('GROQ_API_KEY')
payload = {
    'model': 'openai/gpt-oss-20b',
    'messages': [
        {'role': 'system', 'content': 'You are DocDrift, a precise CI documentation verifier. Given a documentation snippet and a code change, classify whether the documentation is: "still correct", "likely outdated", or "uncertain". Respond ONLY with valid JSON: {"verdict": "<still correct|likely outdated|uncertain>", "reason": "<one-line explanation under 120 chars>", "confidence": <float 0.0 to 1.0>}.'},
        {'role': 'user', 'content': 'DOC: Parameter.human_readable_name is deprecated. CODE CHANGE: renamed to spec. Return JSON with keys: verdict, reason, confidence.'}
    ],
    'temperature': 0.0,
    'max_tokens': 200,
}
req = urllib.request.Request(
    'https://api.groq.com/openai/v1/chat/completions',
    data=json.dumps(payload).encode('utf-8'),
   headers={
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {key}',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) DocDrift/1.0'
},
    method='POST'
)
with urllib.request.urlopen(req, timeout=20) as resp:
    body = json.loads(resp.read().decode())
print(json.dumps(body, indent=2))