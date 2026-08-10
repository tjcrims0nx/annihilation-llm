import json
import sys
from pathlib import Path

# Usage: python analyze_checkpoint.py [path/to/checkpoint.jsonl]
# Defaults to the single checkpoint in the repo's checkpoints/ directory
# when exactly one is present.
checkpoint_dir = Path(__file__).parent / "checkpoints"

if len(sys.argv) > 1:
    checkpoint_path = Path(sys.argv[1])
else:
    candidates = sorted(checkpoint_dir.glob("*.jsonl"))
    if len(candidates) != 1:
        print(f'Found {len(candidates)} checkpoints in {checkpoint_dir}.')
        print('Pass one explicitly: python analyze_checkpoint.py <checkpoint.jsonl>')
        for candidate in candidates:
            print(f'  {candidate.name}')
        sys.exit(1)
    checkpoint_path = candidates[0]

if not checkpoint_path.exists():
    print(f'Checkpoint not found: {checkpoint_path}')
    sys.exit(1)

print(f'Checkpoint: {checkpoint_path}')
print(f'File size: {checkpoint_path.stat().st_size / 1024 / 1024:.2f} MB')

# Read a few trials to understand structure
trials_found = 0
with open(checkpoint_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if data.get('op_code') == 8:
            trials_found += 1
            if trials_found <= 3:
                print(f'\nTrial {i} data keys: {list(data.keys())}')
                if 'user_attr' in data:
                    print(f'  user_attr: {list(data["user_attr"].keys())}')
                    if 'parameters' in data['user_attr']:
                        print(f'  parameters keys: {list(data["user_attr"]["parameters"].keys())}')

print(f'\nTotal trials found: {trials_found}')