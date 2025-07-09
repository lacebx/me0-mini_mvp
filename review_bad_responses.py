import json
import os

BAD_RESPONSES_FILE = 'logs/bad_responses.jsonl'
FINETUNE_PAIRS_FILE = 'logs/finetune_pairs.jsonl'

def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, 'r') as f:
        return [json.loads(line) for line in f if line.strip()]

def save_jsonl(path, data):
    with open(path, 'a') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

def main():
    bad_responses = load_jsonl(BAD_RESPONSES_FILE)
    reviewed = load_jsonl(FINETUNE_PAIRS_FILE)
    reviewed_set = set((r['prompt'], r['bad_response']) for r in reviewed)
    new_pairs = []
    for entry in bad_responses:
        key = (entry['prompt'], entry['bad_response'])
        if key in reviewed_set:
            continue  # Already reviewed
        print('\n---')
        print(f"Prompt: {entry['prompt']}")
        print(f"Bad response: {entry['bad_response']}")
        print(f"Flagged by: {entry['flagged_by']} at {entry['timestamp']}")
        ideal = input("Enter the ideal response (or leave blank to skip): ").strip()
        if ideal:
            new_pairs.append({
                'prompt': entry['prompt'],
                'response': ideal
            })
    if new_pairs:
        save_jsonl(FINETUNE_PAIRS_FILE, new_pairs)
        print(f"\nSaved {len(new_pairs)} new fine-tuning pairs.")
    else:
        print("\nNo new pairs reviewed.")

if __name__ == '__main__':
    main() 