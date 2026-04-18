
import os
import json
from collections import defaultdict
import argparse

def load_data(file_path):
    """Load json data from file."""
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_single_pair_distribution(dataset_dir, output_file):
    """
    Analyze emotion category distribution for documents with exactly one emotion-cause pair.
    """
    stats_buffer = []
    
    def log(msg):
        print(msg)
        stats_buffer.append(msg)

    log(f"\nAnalyzing dataset: {dataset_dir}")
    log("=" * 60)
    
    if not os.path.exists(dataset_dir):
        log(f"Error: Directory not found: {dataset_dir}")
        return

    # Specific files requested
    target_files = ['fold1_test.json', 'fold1_train.json', 'fold1_unlabeled.json', 'fold1_val.json']
    
    directory_stats = {'count': 0, 'single_pair_count': 0, 'emotions': defaultdict(int)}

    for target_file in target_files:
        json_path = os.path.join(dataset_dir, target_file)
        if not os.path.exists(json_path):
            log(f"Skipping {target_file} (not found)")
            continue
            
        data = load_data(json_path)
        file_stats = {'count': 0, 'single_pair_count': 0, 'emotions': defaultdict(int)}
        
        for doc in data:
            file_stats['count'] += 1
            pairs = doc.get('pairs', [])
            
            # Check for exactly one pair
            if len(pairs) == 1:
                file_stats['single_pair_count'] += 1
                
                # Get emotion category
                # Pair format: [emotion_clause_id, cause_clause_id] (integers)
                # Need to find corresponding clause in 'clauses' list
                emotion_clause_id = pairs[0][0]
                
                emotion = 'unknown'
                clauses = doc.get('clauses', [])
                for clause in clauses:
                    # Clause id in dictionary is usually string "1", "2", etc.
                    cid = clause.get('clause_id')
                    if str(cid) == str(emotion_clause_id):
                        emotion = clause.get('emotion_category', 'unknown')
                        break
                
                file_stats['emotions'][emotion] += 1
        
        # Aggregate to directory stats
        directory_stats['count'] += file_stats['count']
        directory_stats['single_pair_count'] += file_stats['single_pair_count']
        for k, v in file_stats['emotions'].items():
            directory_stats['emotions'][k] += v
            
        log(f"File: {target_file}")
        log(f"  Total Docs: {file_stats['count']}")
        percentage = (file_stats['single_pair_count'] / file_stats['count'] * 100) if file_stats['count'] > 0 else 0
        log(f"  Single-Pair Docs: {file_stats['single_pair_count']} ({percentage:.1f}%)")
        log("  Emotion Distribution (Single Pair Only):")
        for emo, count in sorted(file_stats['emotions'].items(), key=lambda x: x[1], reverse=True):
            log(f"    {emo}: {count}")
        log("-" * 40)

    # Print Summary for the directory
    log(f"Summary for {os.path.basename(dataset_dir)}")
    log(f"  Total Docs Aggregated: {directory_stats['count']}")
    total_percentage = (directory_stats['single_pair_count'] / directory_stats['count'] * 100) if directory_stats['count'] > 0 else 0
    log(f"  Total Single-Pair Docs: {directory_stats['single_pair_count']} ({total_percentage:.1f}%)")
    log("  Overall Emotion Distribution (Single Pair Only):")
    
    # Calculate total emotions to get percentages relative to single-pair docs
    total_emotions_count = sum(directory_stats['emotions'].values())
    
    for emo, count in sorted(directory_stats['emotions'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_emotions_count * 100) if total_emotions_count > 0 else 0
        log(f"    {emo}: {count} ({percentage:.1f}%)")
    log("\n")
    
    # Write buffer to file
    with open(output_file, 'a', encoding='utf-8') as f:
        for line in stats_buffer:
            f.write(line + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze single-pair document emotion distribution.')
    parser.add_argument('--dirs', nargs='+', default=[
        'split10_home_train9_test1_disjoint',
        'split10_home_train1_test1_val1_unlabeled7_disjoint'
    ], help='List of dataset directories to analyze')
    parser.add_argument('--output', type=str, default='single_pair_analysis_results.txt', help='Output file path')
    
    args = parser.parse_args()
    
    # Clear output file first
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write("Analysis Results:\n")
    
    for d in args.dirs:
        # Try finding the directory
        target_dir = d
        if not os.path.exists(target_dir):
            if os.path.exists(os.path.join('data', d)):
                target_dir = os.path.join('data', d)
            elif os.path.exists(os.path.join(os.getcwd(), d)):
                target_dir = os.path.join(os.getcwd(), d)
        
        analyze_single_pair_distribution(target_dir, args.output)

