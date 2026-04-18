
import os
import re

base_dir = r'd:\GithubRepo\UECA_ST\ep_split10_t1te1v1_u7_disjoint'

all_files = []
for root, dirs, files in os.walk(base_dir):
    for f in files:
        if f == 'pseudo_label_accuracy_summary.txt':
            all_files.append(os.path.join(root, f))

results = {}

for fpath in sorted(all_files):
    folder = os.path.basename(os.path.dirname(fpath))
    with open(fpath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    round_totals = {}
    overall_error_rate = None
    total_selected = None
    total_error_docs = None
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if '輪總計' in line_stripped and i > 3:
            parts = re.split(r'\s+', line_stripped)
            for j, p in enumerate(parts[1:], 1):
                m = re.match(r'(\d+)/(\d+)', p)
                if m:
                    round_num = f'R{j}'
                    round_totals[round_num] = {'error': int(m.group(1)), 'total': int(m.group(2))}
        if '錯誤率:' in line_stripped:
            m = re.search(r'(\d+\.\d+)%', line_stripped)
            if m:
                overall_error_rate = float(m.group(1))
        if '總選中文檔數:' in line_stripped:
            m = re.search(r'(\d+)', line_stripped)
            if m:
                total_selected = int(m.group(1))
        if '錯誤文檔數:' in line_stripped:
            m = re.search(r'(\d+)', line_stripped)
            if m:
                total_error_docs = int(m.group(1))
    
    results[folder] = {
        'round_totals': round_totals,
        'overall_error_rate': overall_error_rate,
        'total_selected': total_selected,
        'total_error_docs': total_error_docs,
    }

for folder, data in sorted(results.items()):
    print(f'Folder: {folder}')
    print(f'  總錯誤率: {data["overall_error_rate"]}%  (總選: {data["total_selected"]}, 總錯: {data["total_error_docs"]})')
    r10 = data['round_totals'].get('R10', {})
    if r10:
        r10_rate = r10['error']/r10['total']*100 if r10.get('total') else None
        print(f'  R10: {r10.get("error","N/A")}/{r10.get("total","N/A")} = {r10_rate:.1f}%')
    print()
