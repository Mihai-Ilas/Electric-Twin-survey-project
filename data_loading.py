import pickle

def load_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def load_ground_truth(file_path):
    gt_map = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Read the entire file content
            content = f.read()
            
            # Split by comma to get each "Question;Tag" pair
            pairs = content.split(',')
            
            for pair in pairs:
                if ';' in pair:
                    # Split by semicolon to separate Question from Tag
                    parts = pair.split(';')
                    if len(parts) == 2:
                        question = parts[0].strip().lower()
                        tag = parts[1].strip()
                        gt_map[question] = tag         
        return gt_map
    
    except FileNotFoundError:
        print(f"Warning: Ground truth file {file_path} not found.")
        return {}