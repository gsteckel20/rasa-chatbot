import sys
import json
import os
from datetime import datetime

if len(sys.argv) < 2:
    print("Usage: python view_groupme_thread.py <input_json>")
    sys.exit(1)

input_path = sys.argv[1]
output_path = os.path.join('groupme_data', 'thread_plaintext.txt')

with open(input_path, 'r', encoding='utf-8') as f:
    messages = json.load(f)

# Remove all messages where sender is 'GroupMe' or 'system'
messages = [msg for msg in messages if msg.get('name', '').lower() not in ['groupme', 'system'] and msg.get('sender_type', '').lower() != 'system']

# Sort messages by created_at (oldest first)
messages.sort(key=lambda msg: msg.get('created_at', 0))

def format_time(ts):
    try:
        return datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        return 'UnknownTime'

with open(output_path, 'w', encoding='utf-8') as out:
    for msg in messages:
        sender = msg.get('name', 'Unknown')
        text = msg.get('text', '')
        ts = msg.get('created_at', 0)
        time_str = format_time(ts)
        out.write(f'{sender} [{time_str}]: "{text}"\n')

print(f'Saved thread to {output_path}') 