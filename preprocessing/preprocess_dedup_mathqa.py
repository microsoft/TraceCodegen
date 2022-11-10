import json

train_file = './data/mathqa/train_dedup.jsonl'
val_file = './data/mathqa/val_dedup.jsonl'

def rename_task_id(file_name: str, set_name: str):
    with open(file_name, 'r') as f:
        instances = [json.loads(line) for line in f]

    for i, instance in enumerate(instances):
        instance['task_id'] = f"{set_name}_{i}"

    with open(file_name, 'w') as f:
        for instance in instances:
            f.write(json.dumps(instance) + '\n')

if __name__ == "__main__":
    rename_task_id(train_file, 'train')
    rename_task_id(val_file, 'val')