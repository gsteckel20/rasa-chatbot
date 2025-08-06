import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime, timedelta

def parse_timestamp(ts):
    if isinstance(ts, int):
        return datetime.utcfromtimestamp(ts)
    elif isinstance(ts, str) and ts.isdigit():
        return datetime.utcfromtimestamp(int(ts))
    else:
        try:
            return datetime.fromisoformat(ts)
        except Exception:
            return None

def load_and_clean_groupme(input_dir: str) -> List[Dict[str, Any]]:
    input_path = Path(input_dir)
    all_msgs = []
    for json_file in input_path.glob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                messages = json.load(f)
            except Exception:
                continue
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                if msg.get('system'):
                    continue
                text = msg.get('text')
                if not isinstance(text, str) or not text.strip():
                    continue
                group_id = msg.get('group_id')
                created_at = parse_timestamp(msg.get('created_at'))
                if not group_id or not created_at:
                    continue
                all_msgs.append({
                    'msg': msg,
                    'group_id': group_id,
                    'created_at': created_at,
                    'sender': msg.get('name', 'Unknown'),
                    'text': text.strip(),
                    'source': str(json_file)
                })
    # Group by group_id
    from collections import defaultdict
    grouped = defaultdict(list)
    for m in all_msgs:
        grouped[m['group_id']].append(m)
    # For each group, sort by created_at and chunk by 20 min gap
    chunks = []
    gap = timedelta(minutes=300)
    for group_id, msgs in grouped.items():
        msgs.sort(key=lambda x: x['created_at'])
        chunk = []
        chunk_start = None
        chunk_end = None
        chunk_participants = set()
        chunk_raw = []
        chunk_source = None
        for i, m in enumerate(msgs):
            if not chunk:
                # Start new chunk
                chunk = [m]
                chunk_start = m['created_at']
                chunk_end = m['created_at']
                chunk_participants = {m['sender']}
                chunk_raw = [m['msg']]
                chunk_source = m['source']
            else:
                time_gap = m['created_at'] - chunk_end
                if time_gap <= gap:
                    chunk.append(m)
                    chunk_end = m['created_at']
                    chunk_participants.add(m['sender'])
                    chunk_raw.append(m['msg'])
                else:
                    # Save current chunk
                    if chunk_start and chunk_end:
                        cleaned_text = ' '.join([f"{x['sender']}: {x['text']}" for x in chunk])
                        chunks.append({
                            'cleaned_text': cleaned_text,
                            'source': chunk_source,
                            'raw': chunk_raw,
                            'group_id': group_id,
                            'start_time': chunk_start.isoformat() + 'Z',
                            'end_time': chunk_end.isoformat() + 'Z',
                            'participants': list(chunk_participants)
                        })
                    # Start new chunk
                    chunk = [m]
                    chunk_start = m['created_at']
                    chunk_end = m['created_at']
                    chunk_participants = {m['sender']}
                    chunk_raw = [m['msg']]
                    chunk_source = m['source']
        # Save last chunk
        if chunk and chunk_start and chunk_end:
            cleaned_text = ' '.join([f"{x['sender']}: {x['text']}" for x in chunk])
            chunks.append({
                'cleaned_text': cleaned_text,
                'source': chunk_source,
                'raw': chunk_raw,
                'group_id': group_id,
                'start_time': chunk_start.isoformat() + 'Z',
                'end_time': chunk_end.isoformat() + 'Z',
                'participants': list(chunk_participants)
            })
    return chunks

if __name__ == "__main__":
    import sys
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "groupme_data"
    cleaned = load_and_clean_groupme(input_dir)
    print(f"Loaded and cleaned {len(cleaned)} GroupMe message chunks from {input_dir}.") 