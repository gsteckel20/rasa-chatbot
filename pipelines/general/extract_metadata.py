from typing import List, Dict, Any
import re
from datetime import datetime
import os
import uuid

def extract_metadata(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = []
    for idx, item in enumerate(items):
        raw = item.get('raw', {})
        cleaned_text = item.get('cleaned_text', '')
        # Handle if raw is a list (e.g., GroupMe chunk)
        if isinstance(raw, list) and raw:
            first_raw = raw[0]
        elif isinstance(raw, dict):
            first_raw = raw
        else:
            first_raw = {}
        # --- ID ---
        id_val = (
            first_raw.get('id') or
            first_raw.get('course_id') or
            first_raw.get('name') or
            first_raw.get('profile_url') or
            f"chunk-{idx}"
        )
        # --- Source ---
        source = first_raw.get('url') or item.get('source') or ''
        # --- Source File ---
        source_file = os.path.basename(item.get('source', ''))
        # --- Source Type & Content Type ---
        if 'course_description' in first_raw or 'course_id' in first_raw:
            source_type = 'course_catalog'
            content_type = 'course_description'
        elif 'reviews' in first_raw:
            source_type = 'professor_review'
            content_type = 'professor_review'
        elif (isinstance(raw, list) and 'group_id' in item) or (first_raw.get('platform') == 'gm'):
            source_type = 'groupme_message'
            content_type = 'group_chat'
        else:
            source_type = 'department_page'
            content_type = 'general'
        # --- Department ---
        department = first_raw.get('department') or item.get('department') or ''
        # --- Platform ---
        platform = first_raw.get('platform') or item.get('platform') or ''
        # --- Created At ---
        created_at = None
        if 'created_at' in first_raw:
            try:
                ts = first_raw['created_at']
                if isinstance(ts, int):
                    created_at = datetime.utcfromtimestamp(ts).isoformat() + 'Z'
                elif isinstance(ts, str) and ts.isdigit():
                    created_at = datetime.utcfromtimestamp(int(ts)).isoformat() + 'Z'
                else:
                    created_at = str(ts)
            except Exception:
                created_at = str(first_raw['created_at'])
        elif 'start_time' in item:
            created_at = item['start_time']
        # --- Course ID, Professor Name, Group ID, Sender ---
        course_id = first_raw.get('course_id') if 'course_id' in first_raw else None
        professor_name = first_raw.get('name') if source_type == 'professor_review' else None
        group_id = item.get('group_id') or first_raw.get('group_id') if source_type == 'groupme_message' else None
        sender = first_raw.get('name') if source_type == 'groupme_message' else None
        # --- Chunk-level fields for GroupMe ---
        end_time = item.get('end_time') if source_type == 'groupme_message' else None
        participants = item.get('participants') if source_type == 'groupme_message' else None
        # --- Compose metadata dict ---
        metadata = {
            'id': str(uuid.uuid4()),
            'custom_id': f"{source_type}-{id_val}",
            'source': source,
            'source_file': source_file,
            'source_type': source_type,
            'content_type': content_type,
        }
        if department:
            metadata['department'] = department
        if platform:
            metadata['platform'] = platform
        if created_at:
            metadata['created_at'] = created_at
        if end_time:
            metadata['end_time'] = end_time
        if course_id:
            metadata['course_id'] = course_id
        if professor_name:
            metadata['professor_name'] = professor_name
        if group_id:
            metadata['group_id'] = group_id
        if sender:
            metadata['sender'] = sender
        if participants:
            metadata['participants'] = participants
        results.append({
            'text': cleaned_text,
            'metadata': metadata
        })
    return results

if __name__ == "__main__":
    import sys, json
    input_path = sys.argv[1]
    with open(input_path, 'r', encoding='utf-8') as f:
        items = json.load(f)
    meta_items = extract_metadata(items)
    print(f"Extracted metadata for {len(meta_items)} items.") 