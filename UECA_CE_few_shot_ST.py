
import os
import sys
import torch
import argparse
import torch.nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import BertTokenizer, BertForMaskedLM
import time
import numpy as np
import json
import datetime
# from transformers.generation.configuration_utils import CompileConfig


"""setting agrparse"""
parser = argparse.ArgumentParser(description='Training')

"""model struct"""
parser.add_argument('--n_hidden', type=int, default=100, help='number of hidden unit')
parser.add_argument('--n_class', type=int, default=2, help='number of distinct class')
parser.add_argument('--window_size', type=int, default=2, help='size of the emotion cause pair window')
parser.add_argument('--feature_layer', type=int, default=3, help='number of layer iterations')
parser.add_argument('--log_file_name', type=str, default='log', help='name of log file')
parser.add_argument('--model_type', type=str, default='ISML', help='type of model')
"""training"""
parser.add_argument('--training_iter', type=int, default=60, help='number of train iterator')
parser.add_argument('--scope', type=str, default='Ind_BiLSTM', help='scope')
parser.add_argument('--batch_size', type=int, default=8, help='number of example per batch')
parser.add_argument('--learning_rate', type=float, default=0.00001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay for bert')
parser.add_argument('--usegpu', type=bool, default=True, help='gpu')
"""other"""
parser.add_argument('--test_only', type=bool, default=False, help='no training')
parser.add_argument('--checkpoint', type=bool, default=False, help='load checkpoint')
parser.add_argument('--checkpointpath', type=str, default='checkpoint/ECPE/', help='path to load checkpoint')
parser.add_argument('--skip_initial_training', type=bool, default=False, help='skip initial supervised training and go directly to self-training')
parser.add_argument('--savecheckpoint', type=bool, default=True, help='save checkpoint')
parser.add_argument('--save_path', type=str, default='prompt_ECPE_few_shot_ST', help='path to save checkpoint')
parser.add_argument('--experiment_output_dir', type=str, default=None, help='directory to save test results with experiment parameters in name')
parser.add_argument('--device', type=str, default='0', help='device id')
parser.add_argument('--dataset', type=str, default='split10_few_shot_ST/', help='path for dataset')
parser.add_argument('--start_fold', type=int, default=1)
parser.add_argument('--end_fold',   type=int, default=3)
"""self-training parameters"""
parser.add_argument('--threshold', type=float, default=0.9, help='confidence threshold for pseudo-labeling')
parser.add_argument('--gamma', type=float, default=1.0, help='weight for pseudo-labeled loss')
parser.add_argument('--lambda_reg', type=float, default=1e-4, help='L2 regularization weight')
parser.add_argument('--self_training_rounds', type=int, default=10,help='number of self-training rounds')

opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.device

# 動態生成實驗資料夾名稱
def generate_experiment_folder_name(opt, bert_path):
    """根據實驗參數生成資料夾名稱"""
    # 格式化學習率（避免科學記號造成的問題）
    lr_str = f"{opt.learning_rate:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    
    # 格式化 lambda_reg
    lambda_str = f"{opt.lambda_reg:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    
    # 生成時間戳記 (YYYY_MM_DD_HH_MM_SS)
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
    # 從 bert_path 擷取模型名稱
    model_name = os.path.basename(bert_path.rstrip('/'))  # 移除末尾的斜線並取得最後一個路徑部分
    
    folder_name = (
        f"UECA-CE_ST_{timestamp}_f{opt.start_fold}-{opt.end_fold}_"
        f"i{opt.training_iter}_lr{lr_str}_bs{opt.batch_size}_"
        f"wd{opt.weight_decay}_"
        f"{model_name}_th{opt.threshold}_"
        f"gamma{opt.gamma}_reg{lambda_str}_st{opt.self_training_rounds}"
    )
    return folder_name

if opt.usegpu and torch.cuda.is_available():
    use_gpu = True


def print_time():
    print('\n----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime())))


class MyDataset(Dataset):
    def __init__(self, input_file, test=False, tokenizer=None):
        print('load data_file: {}'.format(input_file))
        self.x_bert, self.y_bert, self.label, self.mask_label = [], [], [], []
        self.gt_emotion, self.gt_cause, self.gt_pair = [], [], []
        self.doc_id = []
        self.test = test
        self.n_cut = 0
        self.tokenizer = tokenizer
        cnt_over_limit = 0
        with open(input_file, 'r', encoding='utf8') as f:
            data = json.load(f)

        for doc in data:
            doc_id = doc["doc_id"]
            self.doc_id.append(doc_id)
            d_len = doc["doc_len"]
            pairs = doc["pairs"]
            pos, cause = zip(*pairs) if pairs else ([], [])
            pairs = [tuple(pair) for pair in pairs]  # 將每個子列表轉換為元組

            full_document = ""
            mask_full_document = ""
            mask_label_full_document = ""
            part_sentence = []
            emotions = []
            cnt_emotion_gt = 0
            cnt_cause_gt = 0
            cnt_pair_gt = 0

            # 處理每個子句
            for clause in doc["clauses"]:
                emotion = clause["emotion_category"].strip()
                emotions.append(emotion)
                part_sentence.append(clause["clause"])

            cnt_emotion_gt = len(set(pos))
            cnt_cause_gt = len(set(cause))
            cnt_pair_gt = len(set(pairs))
            self.gt_emotion.append(cnt_emotion_gt)
            self.gt_pair.append(cnt_pair_gt)
            self.gt_cause.append(cnt_cause_gt)
            for i in range(1, d_len + 1):
                full_document = full_document + ' ' + str(i) + ' ' + part_sentence[i - 1]
                mask_full_document = mask_full_document + ' ' + str(i) + ' ' + part_sentence[i - 1]
                mask_label_full_document = mask_label_full_document + ' ' + str(i) + ' ' + part_sentence[i - 1]
                if i in pos:
                    full_document = full_document + '是 '
                    if i in cause:
                        full_document = full_document + '是 '
                        full_document = full_document + ' ' + str(pos[cause.index(i)]) + ' '
                    else:
                        full_document = full_document + '非 '
                        full_document = full_document + ' 无 '
                else:
                    full_document = full_document + '非 '
                    if i in cause:
                        full_document = full_document + '是 '
                        full_document = full_document + ' ' + str(pos[cause.index(i)]) + ' '
                    else:
                        full_document = full_document + '非 '
                        full_document = full_document + ' 无 '

                full_document = full_document + '[SEP]'
                mask_full_document = mask_full_document + "[MASK] [MASK] [MASK] [SEP]"
                mask_label_full_document = mask_label_full_document + "[MASK] [MASK] [MASK] [SEP]"
            if (self.tokenizer.encode_plus(mask_full_document, return_tensors="pt")['input_ids'][0].shape !=
                    self.tokenizer.encode_plus(mask_full_document, return_tensors="pt")['input_ids'][0].shape):
                print('length wrong')

            count_len = len(self.tokenizer.encode_plus(mask_full_document, return_tensors="pt")['input_ids'][0])
            if count_len > 512:
                print("Over limit length{} document{}".format(count_len, doc_id[0]))
                cnt_over_limit += 1
            mask_full_document = \
                self.tokenizer.encode_plus(mask_full_document, return_tensors="pt", max_length=512, truncation=True,
                                           pad_to_max_length=True)['input_ids']
            full_document = \
                self.tokenizer.encode_plus(full_document, return_tensors="pt", max_length=512, truncation=True,
                                           pad_to_max_length=True)['input_ids']
            mask_label_full_document = \
                self.tokenizer.encode_plus(mask_label_full_document, return_tensors="pt", max_length=512,
                                           truncation=True,
                                           pad_to_max_length=True)['input_ids']
            labels = full_document.masked_fill(mask_full_document != 103, -100)
            mask_labels = full_document.masked_fill(mask_label_full_document != 103, -100)

            self.x_bert.append(np.array(mask_full_document[0]))
            self.y_bert.append(np.array(full_document[0]))
            self.label.append(np.array(labels[0]))
            self.mask_label.append(np.array(mask_labels[0]))
        self.x_bert, self.y_bert, self.label, self.mask_label = map(np.array, [self.x_bert, self.y_bert, self.label,
                                                                               self.mask_label])
        self.gt_emotion, self.gt_cause, self.gt_pair = map(np.array, [self.gt_emotion, self.gt_cause, self.gt_pair])
        for var in ['self.x_bert', 'self.y_bert', 'self.label', 'self.mask_label', 'self.gt_emotion', 'self.gt_cause',
                    'self.gt_pair']:
            print('{}.shape {}'.format(var, eval(var).shape))
        print('n_cut {}'.format(self.n_cut))
        print('load data done!\n')

        self.index = [i for i in range(len(self.x_bert))]
        print("num_for_over_limit{}".format(cnt_over_limit))

    def __getitem__(self, index):
        index = self.index[index]
        feed_list = [self.x_bert[index], self.y_bert[index], self.label[index], self.mask_label[index],
                     self.gt_emotion[index], self.gt_cause[index], self.gt_pair[index], True]
        return feed_list

    def __len__(self):
        return len(self.x_bert)




class UnlabeledDataset(Dataset):
    def __init__(self, input_file, tokenizer=None):
        print('load unlabeled data_file: {}'.format(input_file))
        self.x_bert = []
        self.doc_id = []
        self.tokenizer = tokenizer
        with open(input_file, 'r', encoding='utf8') as f:
            data = json.load(f)
        for doc in data:
            doc_id = doc["doc_id"]
            self.doc_id.append(doc_id)
            d_len = doc["doc_len"]
            part_sentence = [clause["clause"] for clause in doc["clauses"]]
            mask_full_document = ""
            for i in range(1, d_len + 1):
                mask_full_document = mask_full_document + ' ' + str(i) + ' ' + part_sentence[i - 1]
                mask_full_document = mask_full_document + "[MASK] [MASK] [MASK] [SEP]"
            mask_full_document = self.tokenizer.encode_plus(mask_full_document, return_tensors="pt", max_length=512, truncation=True, pad_to_max_length=True)['input_ids']
            self.x_bert.append(np.array(mask_full_document[0]))
        self.x_bert = np.array(self.x_bert)
        self.index = [i for i in range(len(self.x_bert))]
        print('load unlabeled data done!\n')
    def __getitem__(self, index):
        index = self.index[index]
        return self.x_bert[index]
    def __len__(self):
        return len(self.x_bert)
    




class PseudoLabeledDataset(Dataset):
    def __init__(self, pseudo_labeled_samples, doc_ids, seq_len=512, mask_token_id=103):
        self.x_bert = [x for x, y in pseudo_labeled_samples]  # input_ids (masked)
        self.pseudo_labels = [y for x, y in pseudo_labeled_samples]  # 只包含 [MASK] 位置的 token id
        self.doc_ids = doc_ids
        self.seq_len = seq_len
        self.mask_token_id = mask_token_id

    def __getitem__(self, index):
        x = self.x_bert[index]  # 保持為 numpy array
        # 構造 y_bert, label, mask_label
        y = np.zeros(self.seq_len, dtype=np.int64)  # 使用 numpy 而不是 torch
        label = np.full(self.seq_len, -100, dtype=np.int64)
        mask_label = np.full(self.seq_len, -100, dtype=np.int64)

        # 找出 [MASK] 位置，把 pseudo_label 填進去
        mask_positions = np.where(x == self.mask_token_id)[0]
        pseudo_label = self.pseudo_labels[index]
        y[mask_positions] = pseudo_label
        label[mask_positions] = pseudo_label
        mask_label[mask_positions] = pseudo_label

        gt_emotion = np.int64(0)
        gt_cause = np.int64(0)
        gt_pair = np.int64(0)
        return [x, y, label, mask_label, gt_emotion, gt_cause, gt_pair, False]

    def __len__(self):
        return len(self.x_bert)


























class prompt_bert(torch.nn.Module):
    def __init__(self, bert_path='./bert-base-chinese'):
        super(prompt_bert, self).__init__()
        self.bert = BertForMaskedLM.from_pretrained(bert_path)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.bert.resize_token_embeddings(len(self.tokenizer))

    def forward(self, x_bert, labels):
        output = self.bert(x_bert, labels=labels)
        loss, logits = output.loss, output.logits
        return loss, logits


def print_training_info():
    print('\n\n>>>>>>>>>>>>>>>>>>>>TRAINING INFO:\n')
    print('batch-{}, lr-{}'.format(
        opt.batch_size, opt.learning_rate))
    print('training_iter-{}\n'.format(opt.training_iter))


def crf_prompt(logits, labels, x_bert, gt_emotion, gt_cause, gt_pair, save_path="results.txt"):
    label_index = [122, 123, 124, 125, 126, 127, 128, 129, 130, 8108, 8111, 8110, 8124, 8122, 8115, 8121, 8126, 8123,
                   8131, 8113, 8128, 8130, 8133, 8125, 8132, 8153, 8149, 8143, 8162, 8114, 8176, 8211, 8226, 8229, 8198,
                   8216, 8234, 8218, 8240, 8164, 8245, 8239, 8250, 8252, 8208, 8248, 8264, 8214, 8249, 8145, 8246, 8247,
                   8251, 8267, 8222, 8259, 8272, 8255, 8257, 8183, 8398, 8356, 8381, 8308, 8284, 8347, 8369, 8360, 8419,
                   8203, 8459, 8325, 8454, 8473, 8273] # 表示數字1-75的token id
    emo_gt = torch.sum(gt_emotion)
    emo_pre = 0
    emo_acc = 0
    cause_gt = torch.sum(gt_cause)
    cause_pre = 0
    cause_acc = 0
    pair_gt = torch.sum(gt_pair)
    pair_pre = 0
    pair_acc = 0
    for i in range(labels.shape[0]):
        count_mask = -1
        j = 0
        count_sentence = 0
        while j < 512:
            if x_bert[i][j] == 103:     #當token為'[MASK]'
                count_mask += 1
                count_mask = count_mask % 3 # 每三個[MASK]就重新從0開始
                if count_mask == 0:     #第一個'[MASK]'(預測是否為情緒子句)
                    if labels[i][j] == 3221:        #真實label為'是'
                        if torch.argmax(logits[i][j]) == 3221:      #預測為'是'
                            emo_acc += 1                            #預測正確的情緒數量+1
                    if torch.argmax(logits[i][j]) == 3221:         #預測為'是'
                        emo_pre += 1                               #預測為情緒的數量+1

                if count_mask == 1:     #第二個'[MASK]'(預測是否為原因子句)
                    if torch.argmax(logits[i][j]) == 3221:  #預測為'是'
                        cause_pre += 1                      #預測為原因的數量+1
                    if labels[i][j] == 3221:                #真實label為'是'
                        if torch.argmax(logits[i][j]) == 3221:  #預測為'是'
                            cause_acc += 1                      #預測正確的原因數量+1

                if count_mask == 2:   #第三個'[MASK]'(預測相關子句)
                    count_sentence += 1 #計算當前句子數量
                    mask = torch.zeros([21128]) #初始化一個大小為21128的全零張量作為遮罩（對照表的大小）
                    case = [label_index[k] for k in range(max(0, -opt.window_size + count_sentence - 1),
                                                          min(75, opt.window_size + count_sentence))]
                    #建立一個case列表，包含當前句子範圍內的標籤索引，滑動視窗來選取有效的標籤範圍(避免超出句子邊界)
                    case.append(3187)  #加入'无'的token_id
                    for index in case:      #遍歷case，將對應mask設為1(lable_idex為數字1~75)
                        mask[index] = 1
                    logits_ = torch.argmax(logits[i][j] * mask)  #過濾logits，選取最高機率的索引作為預測結果(找到對應1-75數字的機率，然後經過argmax找到最大機率的索引)

                    if logits_ in label_index :  #當預測結果在label_index中(數字1~75)
                        pair_pre += 1           #預測的組合數量+1
                    if labels[i][j] in label_index:  #當真實label在label_index中(數字1~75)
                        if logits_ == labels[i][j] :  #且當預測結果與真實label相同
                            pair_acc += 1               #預測正確的組合數量+1

                j = j + 1
            else:
                j = j + 1
    p_emotion = emo_acc / (emo_pre + 1e-8)
    p_cause = cause_acc / (cause_pre + 1e-8)
    p_pair = pair_acc / (pair_pre + 1e-8)
    r_emotion = emo_acc / (emo_gt + 1e-8)
    r_cause = cause_acc / (cause_gt + 1e-8)
    r_pair = pair_acc / (pair_gt + 1e-8)
    f_emotion = 2 * p_emotion * r_emotion / (p_emotion + r_emotion + 1e-8)
    f_cause = 2 * p_cause * r_cause / (p_cause + r_cause + 1e-8)
    f_pair = 2 * p_pair * r_pair / (p_pair + r_pair + 1e-8)
    with open(save_path, "a", encoding="utf-8") as file:
        file.write(f"Emotion: 預測正確的數量:{emo_acc}    預測出情緒的數量:{emo_pre}    實際正確情緒的數量:{emo_gt}\n")
        file.write(f"Cause:   預測正確的數量:{cause_acc}  預測出原因的數量:{cause_pre}  實際正確原因的數量:{cause_gt}\n")
        file.write(f"Pair:    預測正確的數量:{pair_acc}   預測出組合的數量:{pair_pre}   實際正確組合的數量:{pair_gt}\n")
        file.write(f"Precision (Emotion, Cause, Pair): {p_emotion:.4f}, {p_cause:.4f}, {p_pair:.4f}\n")
        file.write(f"Recall (Emotion, Cause, Pair): {r_emotion:.4f}, {r_cause:.4f}, {r_pair:.4f}\n")
        file.write(f"F1 Score (Emotion, Cause, Pair): {f_emotion:.4f}, {f_cause:.4f}, {f_pair:.4f}\n\n")
    print('emo_gt {}  cause_gt {}  pair_gt {}'.format(emo_gt, cause_gt, pair_gt))
    print(f"Emotion: 預測正確的數量:{emo_acc}, 預測出情緒的數量:{emo_pre}, 實際正確情緒的數量:{emo_gt}")
    print(f"Cause: 預測正確的數量:{cause_acc}, 預測出原因的數量:{cause_pre}, 實際正確原因的數量:{cause_gt}")
    print(f"Pair: 預測正確的數量:{pair_acc}, 預測出組合的數量:{pair_pre}, 實際正確組合的數量:{pair_gt}")

    return p_emotion, r_emotion, f_emotion, p_cause, r_cause, f_cause, p_pair, r_pair, f_pair



def save_mask_predictions(logits, x_bert, tokenizer, doc_ids, fold, output_dir=".", base_filename="text_result", pseudo_labels=None):
    label_index = [122, 123, 124, 125, 126, 127, 128, 129, 130, 8108, 8111, 8110, 8124, 8122, 8115, 8121, 8126, 8123,
                   8131, 8113, 8128, 8130, 8133, 8125, 8132, 8153, 8149, 8143, 8162, 8114, 8176, 8211, 8226, 8229, 8198,
                   8216, 8234, 8218, 8240, 8164, 8245, 8239, 8250, 8252, 8208, 8248, 8264, 8214, 8249, 8145, 8246, 8247,
                   8251, 8267, 8222, 8259, 8272, 8255, 8257, 8183, 8398, 8356, 8381, 8308, 8284, 8347, 8369, 8360, 8419,
                   8203, 8459, 8325, 8454, 8473, 8273]
    output_lines = []
    for i, doc_id in enumerate(doc_ids):
        output_lines.append(f"doc_id :{doc_id}")
        line = []
        count_sentence = 0
        count_mask = -1
        j = 0
        pseudo_idx = 0  # 追蹤 pseudo_label 的位置
        while j < 512:
            if x_bert[i][j] == 103:     #當token為'[MASK]'
                count_mask += 1
                count_mask = count_mask % 3
                if pseudo_labels is not None:
                     pred_token_id = pseudo_labels[i][pseudo_idx]
                     pseudo_idx += 1
                     pred_text = tokenizer.decode([pred_token_id]).strip()
                     line.append(pred_text)
                else:
                    # count_mask == 0 → emotion
                    if count_mask == 0:
                        pred_token_id = torch.argmax(logits[i][j]).item()
                        pred_text = tokenizer.decode([pred_token_id]).strip()
                        line.append(pred_text)
                    
                    # count_mask == 1 → cause
                    elif count_mask == 1:
                        pred_token_id = torch.argmax(logits[i][j]).item()
                        pred_text = tokenizer.decode([pred_token_id]).strip()
                        line.append(pred_text)

                    # count_mask == 2 → pair
                    elif count_mask == 2:
                        count_sentence += 1
                        case = [label_index[k] for k in range(
                            max(0, -opt.window_size + count_sentence - 1),
                            min(75, opt.window_size + count_sentence)
                        )]
                        case.append(3187)
                        candidate_logits = logits[i][j][case]
                        selected_idx = torch.argmax(candidate_logits).item()
                        pred_token_id = case[selected_idx]  
                        pred_text = tokenizer.decode([pred_token_id]).strip()
                        line.append(pred_text)
                if count_mask == 2:
                    # 一組三個 [MASK] 預測完成，寫入一行
                    output_lines.append(' '.join(line))
                    line = []
                    
                j += 1
            else:
                j += 1

    output_file = os.path.join(output_dir, f"fold{fold}_{base_filename}.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(line + "\n")
    print(f"儲存完成，路徑: {output_file}")





def run():
    if opt.log_file_name:
        save_path = opt.save_path

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # sys.stdout = open(save_path + '/' + opt.log_file_name, 'w')

    log_dir = "init_supervised_metrics"
    os.makedirs(log_dir, exist_ok=True)
    
    # 創建時間記錄目錄
    time_log_dir = "time_logs"
    os.makedirs(time_log_dir, exist_ok=True)
    time_log_file = os.path.join(time_log_dir, "execution_time_log.txt")

    print_time()
    overall_start_time = time.time()
    bert_path = './bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(bert_path)

    # train
    print_training_info()  # 輸出訓練的超參數資訊

    max_result_emo_f, max_result_emo_p, max_result_emo_r = [], [], []
    max_result_pair_f, max_result_pair_p, max_result_pair_r = [], [], []
    max_result_cause_f, max_result_cause_p, max_result_cause_r = [], [], []
    
    all_folds_start_time = time.time()  # 記錄所有 fold 開始時間
    
    # 初始化時間記錄文件
    with open(time_log_file, "w", encoding="utf-8") as f:
        f.write("=== UECA Self-Training 執行時間記錄 ===\n")
        f.write(f"實驗開始時間: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overall_start_time))}\n")
        f.write(f"Fold 範圍: {opt.start_fold} - {opt.end_fold}\n")
        f.write(f"Self-training 輪數: {opt.self_training_rounds}\n")
        f.write(f"批次大小: {opt.batch_size}\n")
        f.write(f"學習率: {opt.learning_rate}\n")
        f.write("=" * 50 + "\n\n")
    
    fold_times = []  # 記錄每個fold的時間信息
    
    # 生成時間戳記用於 self_training_results 資料夾命名
    st_timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
    for fold in range(opt.start_fold, opt.end_fold + 1):
        fold_start_time = time.time()  # 記錄當前 fold 開始時間
        print(f"\n  Fold {fold} 開始時間: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        
        # model
        print('build model..')
        model = prompt_bert(bert_path)
        print('build model end...')
        if opt.checkpoint:
            model = torch.load(opt.checkpointpath + '/fold{}.pth'.format(fold),
                               map_location=torch.device('cpu'))
        if use_gpu:
            model = model.cuda()


        train_file_name = 'fold{}_train_few_shot.json'.format(fold)
        val_file_name = 'fold{}_val.json'.format(fold)
        test_file_name = 'fold{}_test.json'.format(fold)
        unlabeled_file_name = 'fold{}_unlabeled.json'.format(fold)
        unlabeled = opt.dataset + unlabeled_file_name


        print('############# fold {} begin ###############'.format(fold))
        train = opt.dataset + train_file_name
        val = opt.dataset + val_file_name
        test = opt.dataset + test_file_name
        edict = {"train": train, "val": val, "test": test}
        unlabeled_dataset = UnlabeledDataset(unlabeled, tokenizer=tokenizer)
        NLP_Dataset = {x: MyDataset(edict[x], test=(x == 'test'), tokenizer=tokenizer) for x in ['train', 'val', 'test']}
        trainloader = DataLoader(NLP_Dataset['train'], batch_size=opt.batch_size, shuffle=True, drop_last=True)
        valloader = DataLoader(NLP_Dataset['val'], batch_size=opt.batch_size, shuffle=False)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=opt.batch_size, shuffle=False)
        """
        num_train_data = len(NLP_Dataset['train'])
        print(f"--- 訓練資料集 (NLP_Dataset['train']) 總共有: {num_train_data} 筆資料") # 1750筆資料
        """
        testloader = DataLoader(NLP_Dataset['test'], batch_size=opt.batch_size, shuffle=False)
        """
        num_test_data = len(NLP_Dataset['test'])
        print(f"--- 訓練資料集 (NLP_Dataset['train']) 總共有: {num_test_data} 筆資料") # 195筆資料
        """
    
        max_p_emotion, max_r_emotion, max_f1_emotion, max_p_cause, max_r_cause, max_f1_cause, max_p_pair,\
        max_r_pair, max_f1_pair = [-1.] * 9
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
        
        # 檢查是否要跳過初始訓練
        if opt.skip_initial_training:
            print("=== 跳過初始監督訓練，直接載入預訓練模型 ===")
            pretrained_model_path = f'prompt_ECPE_few_shot_ST/fold{fold}.pth'
            if os.path.exists(pretrained_model_path):
                model = torch.load(pretrained_model_path, map_location=torch.device('cuda' if use_gpu else 'cpu'))
                if use_gpu:
                    model = model.cuda()
                print(f"已載入預訓練模型: {pretrained_model_path}")
                
                # 直接跳到 self-training 部分
                print("=== 開始 Self-Training 流程 ===")
                os.makedirs("pseudo_results", exist_ok=True)
            else:
                print(f"找不到預訓練模型: {pretrained_model_path}")
                print("請確認模型檔案存在，或設定 skip_initial_training=False")
                continue
                
        elif opt.test_only:
            # 在 test_only 模式下生成實驗資料夾名稱並建立資料夾
            if opt.experiment_output_dir is None:
                opt.experiment_output_dir = generate_experiment_folder_name(opt, bert_path)
                print(f"測試結果將儲存至: {opt.experiment_output_dir}")
            
            # 確保實驗輸出資料夾存在
            os.makedirs(opt.experiment_output_dir, exist_ok=True)
            
            all_test_logits = torch.tensor([])
            all_test_label = torch.tensor([])
            all_test_mask_label = torch.tensor([])
            all_test_y_bert = torch.tensor([])
            all_test_x_bert = torch.tensor([])
            all_test_emotion_gt = torch.tensor([])
            all_test_cause_gt = torch.tensor([])
            all_test_pair_gt = torch.tensor([])
            model.eval()
            
            # 將結果儲存到實驗資料夾而不是 save_path
            output_file_path = os.path.join(opt.experiment_output_dir, 'test_results.txt')
            with torch.no_grad(), open(output_file_path, 'w', encoding='utf-8') as output_file:
                for idx, data in enumerate(testloader):
                    x_bert, y_bert, label, mask_label, gt_emotion, gt_cause, gt_pair, _ = data
                    if use_gpu:
                        x_bert = x_bert.cuda()
                        y_bert = y_bert.cuda()
                        label = label.cuda()
                        mask_label = mask_label.cuda()
                    loss, logits = model(x_bert, label)
                    logits = F.softmax(logits, dim=-1)
                    all_test_label = torch.cat((all_test_label, label.cpu()), 0)
                    all_test_mask_label = torch.cat((all_test_mask_label, mask_label.cpu()), 0)
                    all_test_logits = torch.cat((all_test_logits, logits.cpu()), 0)
                    all_test_y_bert = torch.cat((all_test_y_bert, y_bert.cpu()), 0)
                    all_test_x_bert = torch.cat((all_test_x_bert, x_bert.cpu()), 0)
                    all_test_emotion_gt = torch.cat((all_test_emotion_gt, gt_emotion), 0)
                    all_test_cause_gt = torch.cat((all_test_cause_gt, gt_cause), 0)
                    all_test_pair_gt = torch.cat((all_test_pair_gt, gt_pair), 0)


                p_emotion, r_emotion, f_emotion, p_cause, r_cause, f_cause, p_pair, r_pair, f_pair = crf_prompt(
                    all_test_logits, all_test_label, all_test_x_bert, all_test_emotion_gt, all_test_cause_gt,
                    all_test_pair_gt, save_path=os.path.join(opt.experiment_output_dir, f"test_evaluation_fold{fold}.txt"))
                save_mask_predictions(all_test_logits, all_test_x_bert, tokenizer, NLP_Dataset['test'].doc_id, 
                                    fold=fold, output_dir=opt.experiment_output_dir)
                print(
                    "e_p: {:.4f} e_r: {:.4f} e_f: {:.4f} c_p: {:.4f} c_r: {:.4f} c_f: {:.4f}"
                    " pair_p: {:.4f} pair_r: {:.4f} pair_f: {:.4f}".format(
                        p_emotion,
                        r_emotion,
                        f_emotion,
                        p_cause,
                        r_cause,
                        f_cause,
                        p_pair,
                        r_pair,
                        f_pair))
                if f_emotion > max_f1_emotion:
                    max_f1_emotion, max_p_emotion, max_r_emotion = f_emotion, p_emotion, r_emotion
                if f_cause > max_f1_cause:
                    max_f1_cause, max_p_cause, max_r_cause = f_cause, p_cause, r_cause
                if f_pair > max_f1_pair:
                    max_f1_pair, max_p_pair, max_r_pair = f_pair, p_pair, r_pair

                print(
                    "max result---- e_p: {:.4f} e_r: {:.4f} e_f: {:.4f} c_p: {:.4f} c_r: {:.4f} c_f: {:.4f}"
                    " pair_p: {:.4f} pair_r: {:.4f} pair_f: {:.4f}".format(
                        max_p_emotion, max_r_emotion, max_f1_emotion, max_p_cause, max_r_cause, max_f1_cause,
                        max_p_pair, max_r_pair, max_f1_pair))
                
                # test_only 模式下，記錄時間資訊
                fold_total_time = time.time() - fold_start_time
                print(f"\nFold {fold} (Test-only) 完成！")
                print(f"Fold {fold} 總執行時間: {fold_total_time/60:.1f} 分鐘")
                print(f"Fold {fold} 結束時間: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
                
                # 記錄 test_only 模式的時間信息
                fold_time_info = {
                    'fold': fold,
                    'total_time': fold_total_time,
                    'supervised_time': 0,  # test_only 模式沒有訓練時間
                    'self_training_time': 0,  # test_only 模式沒有 self-training 時間
                    'is_test_only': True  # 標記為測試模式
                }
                fold_times.append(fold_time_info)
                
                # test_only 模式下，完成測試後跳過後續的 self-training
                continue

        else:
            # 添加早停機制
            patience = 10  # 連續10輪沒改善就停止
            patience_counter = 0
            early_stop_iter = -1  # 記錄早停時的 iter 數
            
            supervised_training_start_time = time.time()  # 記錄監督訓練開始時間
            print(f"初始監督訓練開始時間: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
            
            for i in range(opt.training_iter):
                model.train()
                start_time, step = time.time(), 1
                # 此迴圈根據訓練資料集大小決定要跑幾次: 假設175筆資料、batch_size=8，\
                # 則此迴圈會跑21次(175/8=21.875，drop_last=True會捨去最後不足8筆資料) \ 
                # 21 * 8 = 168筆資料，175-168=7筆資料，所以會有7筆資料沒有被訓練到
                for index, data in enumerate(trainloader):
                    with torch.autograd.set_detect_anomaly(True):
                        x_bert, y_bert, label, mask_label, gt_emotion, gt_cause, gt_pair, is_labeled = data
                        if use_gpu:
                            x_bert = x_bert.cuda()
                            y_bert = y_bert.cuda()
                            label = label.cuda()
                            mask_label = mask_label.cuda()



                        loss, logits = model(x_bert, mask_label)
                        logits = F.softmax(logits, dim=-1)


                        optimizer.zero_grad()
                        if use_gpu:
                            loss = loss.cuda()
                        loss.backward()
                        optimizer.step()

                        print("loss: {:.4f}".format(loss))
                        # 每20個批次，執行crf_prompt
                        if index % 20 == 0:
                            p_emotion, r_emotion, f_emotion, p_cause, r_cause, f_cause, p_pair, r_pair, f_pair = \
                                crf_prompt(logits.cpu(), label.cpu(), x_bert.cpu(), gt_emotion, gt_cause, gt_pair, 
                                         save_path=f"training_progress_{st_timestamp}_fold{fold}_iter{i}_batch{index}.txt")
                            print(
                                "iter: {} e_p: {:.4f} e_r: {:.4f} e_f: {:.4f} c_p: {:.4f} c_r: {:.4f} c_f: {:.4f}"
                                " pair_p: {:.4f} pair_r: {:.4f} pair_f: {:.4f}".format(
                                    index,
                                    p_emotion,
                                    r_emotion,
                                    f_emotion,
                                    p_cause,
                                    r_cause,
                                    f_cause,
                                    p_pair,
                                    r_pair,
                                    f_pair))
                all_val_logits = torch.tensor([])
                all_val_label = torch.tensor([])
                all_val_mask_label = torch.tensor([])
                all_val_y_bert = torch.tensor([])
                all_val_x_bert = torch.tensor([])
                all_val_emotion_gt = torch.tensor([])
                all_val_cause_gt = torch.tensor([])
                all_val_pair_gt = torch.tensor([])

                model.eval()
                with torch.no_grad():
                    for _, data in enumerate(valloader):  # 改為驗證集
                        x_bert, y_bert, label, mask_label, gt_emotion, gt_cause, gt_pair, _ = data
                        if use_gpu:
                            x_bert = x_bert.cuda()
                            y_bert = y_bert.cuda()
                            label = label.cuda()
                            mask_label = mask_label.cuda()
                        loss, logits = model(x_bert, label)
                        logits = F.softmax(logits, dim=-1)
                        all_val_label = torch.cat((all_val_label, label.cpu()), 0)
                        all_val_mask_label = torch.cat((all_val_mask_label, mask_label.cpu()), 0)
                        all_val_logits = torch.cat((all_val_logits, logits.cpu()), 0)
                        all_val_y_bert = torch.cat((all_val_y_bert, y_bert.cpu()), 0)
                        all_val_x_bert = torch.cat((all_val_x_bert, x_bert.cpu()), 0)
                        all_val_emotion_gt = torch.cat((all_val_emotion_gt, gt_emotion), 0)
                        all_val_cause_gt = torch.cat((all_val_cause_gt, gt_cause), 0)
                        all_val_pair_gt = torch.cat((all_val_pair_gt, gt_pair), 0)

                    p_emotion, r_emotion, f_emotion, p_cause, r_cause, f_cause, p_pair, r_pair, f_pair = crf_prompt(
                        all_val_logits, all_val_label, all_val_x_bert, all_val_emotion_gt, all_val_cause_gt,
                        all_val_pair_gt, save_path=f"validation_results_{st_timestamp}_fold{fold}_iter{i}.txt")
                    print("iter{} validation result:".format(i))  # 改為驗證結果
                    print(
                        "e_p: {:.4f} e_r: {:.4f} e_f: {:.4f} c_p: {:.4f} c_r: {:.4f} c_f: {:.4f} pair_p: {:.4f}"
                        " pair_r: {:.4f} pair_f: {:.4f}".format(
                            p_emotion,
                            r_emotion,
                            f_emotion,
                            p_cause,
                            r_cause,
                            f_cause,
                            p_pair,
                            r_pair,
                            f_pair))
                    if f_emotion > max_f1_emotion:
                        max_f1_emotion, max_p_emotion, max_r_emotion = f_emotion, p_emotion, r_emotion
                    if f_cause > max_f1_cause:
                        max_f1_cause, max_p_cause, max_r_cause = f_cause, p_cause, r_cause
                    if f_pair > max_f1_pair:
                        max_f1_pair, max_p_pair, max_r_pair = f_pair, p_pair, r_pair
                        patience_counter = 0  # 重設耐心計數器
                        print(f"  新的最佳 F1: {f_pair:.4f} at iter {i}")
                        if opt.savecheckpoint:
                            torch.save(model, save_path + '/' + 'fold{}.pth'.format(fold))
                            print(f"Model for fold {fold} saved to {save_path}/fold{fold}.pth")
                    else:
                        patience_counter += 1
                        print(f"  沒有改善，patience: {patience_counter}/{patience}")
                    
                    if patience_counter >= patience:
                        early_stop_iter = i + 1  # 記錄早停的 iter
                        print(f"  Early stopping at iter {i+1}, best F1: {max_f1_pair:.4f}")
                        break
                    print("iter{} test result:".format(i))
                    print(
                        "max result---- e_p: {:.4f} e_r: {:.4f} e_f: {:.4f} c_p: {:.4f} c_r: {:.4f} c_f: {:.4f}"
                        " pair_p: {:.4f} pair_r: {:.4f} pair_f: {:.4f}".format(
                            max_p_emotion, max_r_emotion, max_f1_emotion, max_p_cause, max_r_cause, max_f1_cause,
                            max_p_pair, max_r_pair, max_f1_pair))
            max_result_emo_f.append(max_f1_emotion)
            max_result_cause_f.append(max_f1_cause)
            max_result_pair_f.append(max_f1_pair)
            max_result_emo_p.append(max_p_emotion)
            max_result_cause_p.append(max_p_cause)
            max_result_pair_p.append(max_p_pair)
            max_result_emo_r.append(max_r_emotion)
            max_result_cause_r.append(max_r_cause)
            max_result_pair_r.append(max_r_pair)

            log_file = os.path.join(log_dir, f"fold{fold}_init_supervised_metrics.txt")
            with open(log_file, "w", encoding="utf-8") as f:
                f.write("Emotion F1: " + str(max_result_emo_f) + "\n")
                f.write("Emotion P: " + str(max_result_emo_p) + "\n")
                f.write("Emotion R: " + str(max_result_emo_r) + "\n")
                f.write("Cause F1: " + str(max_result_cause_f) + "\n")
                f.write("Cause P: " + str(max_result_cause_p) + "\n")
                f.write("Cause R: " + str(max_result_cause_r) + "\n")
                f.write("Pair F1: " + str(max_result_pair_f) + "\n")
                f.write("Pair P: " + str(max_result_pair_p) + "\n")
                f.write("Pair R: " + str(max_result_pair_r) + "\n")
                # 記錄早停資訊
                if early_stop_iter > 0:
                    f.write(f"Early Stopping: Yes (at iter {early_stop_iter})\n")
                else:
                    f.write(f"Early Stopping: No (completed all {opt.training_iter} iters)\n")
                f.write(f"Best F1 achieved: {max_f1_pair:.4f}\n")  # 應該用 max_f1_pair 而不是 best_val_f1

            # ====== 當前第 i 折最佳模型的路徑 ======
            model_path = save_path + '/' + 'fold{}.pth'.format(fold)
            model = torch.load(model_path, map_location=torch.device('cuda' if use_gpu else 'cpu'))
            if use_gpu:
                model = model.cuda()
            print(f"已載入初始化監督訓練的最佳模型: {model_path}")
            
            # 輸出當前 fold 的早停資訊
            supervised_training_total_time = time.time() - supervised_training_start_time
            print(f"  初始監督訓練執行時間: {supervised_training_total_time/60:.1f} 分鐘")
            
            if early_stop_iter > 0:
                print(f"  Fold {fold} - Early stopping at iter {early_stop_iter}, Best F1: {max_f1_pair:.4f}")
            else:
                print(f"  Fold {fold} - Completed all {opt.training_iter} iters, Best F1: {max_f1_pair:.4f}")

        # 如果跳過監督訓練，設定時間為0
        if opt.skip_initial_training:
            supervised_training_total_time = 0

        # ====== Self-Training 開始 ======
        # 無論是否跳過初始訓練，都會執行 self-training
        print("=== 開始 Self-Training 流程 ===")
        os.makedirs("pseudo_results", exist_ok=True)
        
        # 建立 self-training 測試結果專用資料夾，加上時間戳記
        st_results_dir = f"self_training_results_{st_timestamp}"
        os.makedirs(st_results_dir, exist_ok=True)
        
        self_training_start_time = time.time()  # 記錄 self-training 開始時間
        print(f"  Self-training 開始時間: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        
        for self_round in range(opt.self_training_rounds):
            round_start_time = time.time()  # 記錄每輪開始時間
            print(f"=== Self-training round {self_round+1} / {opt.self_training_rounds} ===")
            print(f"  Round {self_round+1} 開始時間: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
            
            # 如果不是第一輪 self-training，載入前一輪的最佳模型
            if self_round > 0:
                st_save_dir = os.path.join(save_path, 'self_training_models')
                prev_best_model_path = os.path.join(st_save_dir, f'fold{fold}_self_training_best.pth')
                if os.path.exists(prev_best_model_path):
                    model = torch.load(prev_best_model_path, map_location=torch.device('cuda' if use_gpu else 'cpu'))
                    if use_gpu:
                        model = model.cuda()
                    print(f"  已載入前一輪最佳模型: {prev_best_model_path}")
                    # 重新初始化優化器，使用新模型的參數
                    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
                else:
                    print(f"  找不到前一輪最佳模型: {prev_best_model_path}，繼續使用當前模型")
            
            with torch.no_grad():
                model.eval()
                all_pseudo_labeled_samples = []
                all_pseudo_doc_ids = []
                all_pseudo_predictions = []  # 收集所有pseudo預測結果用於最後保存
                batch_start_idx = 0

                for batch_x in unlabeled_loader:
                    if use_gpu:
                        batch_x = batch_x.cuda()
                    _, logits = model(batch_x, labels=None)
                    logits = F.softmax(logits, dim=-1)
                    # print("第一個 batch 預測完成，logits shape:", logits.shape)  # 註釋掉詳細輸出

                    mask_token_id = 103  # BERT 的 [MASK] token id
                    threshold = opt.threshold
                    label_index = [122, 123, 124, 125, 126, 127, 128, 129, 130, 8108, 8111, 8110, 8124, 8122, 8115, 8121, 8126, 8123,
                                   8131, 8113, 8128, 8130, 8133, 8125, 8132, 8153, 8149, 8143, 8162, 8114, 8176, 8211, 8226, 8229, 8198,
                                   8216, 8234, 8218, 8240, 8164, 8245, 8239, 8250, 8252, 8208, 8248, 8264, 8214, 8249, 8145, 8246, 8247,
                                   8251, 8267, 8222, 8259, 8272, 8255, 8257, 8183, 8398, 8356, 8381, 8308, 8284, 8347, 8369, 8360, 8419,
                                   8203, 8459, 8325, 8454, 8473, 8273]
                    
                    pseudo_labeled_samples = []
                    pseudo_labeled_indices = []

                    for i in range(batch_x.shape[0]):
                        mask_positions = (batch_x[i] == mask_token_id).nonzero(as_tuple=True)[0] # 找出第 i 個樣本裡所有 [MASK] token 的索引
                        pseudo_label = []
                        all_pass = True
                        
                        # 記錄每個子句的預測結果（用於第三個[MASK]的邏輯判斷）
                        cause_predictions = {}  # {sentence_idx: is_cause}
                        
                        for j, pos in enumerate(mask_positions): # 這個 for 迴圈會跑 len(mask_positions) 輪，也就是 N × 3 輪（N = 子句數)
                            # j: 目前是第幾個 [MASK]
                            # pos: 這個 [MASK] 在 token 序列中的實際索引位置
                            count_sentence = j // 3 + 1  # 這是第幾個子句
                            mask_type = j % 3  # 0:情緒, 1:原因, 2:配對
                            
                            if mask_type == 0:  # 第一個 [MASK] (情緒子句預測)
                                prob, pred_token_id = torch.max(logits[i, pos], dim=-1)
                                if prob.item() >= threshold:
                                    pseudo_label.append(pred_token_id if isinstance(pred_token_id, int) else pred_token_id.item())
                                else:
                                    all_pass = False
                                    break
                            elif mask_type == 1:  # 第二個 [MASK] (原因子句預測)
                                prob, pred_token_id = torch.max(logits[i, pos], dim=-1)
                                if prob.item() >= threshold:
                                    pred_token_id_val = pred_token_id if isinstance(pred_token_id, int) else pred_token_id.item()
                                    pseudo_label.append(pred_token_id_val)
                                    # 記錄當前子句是否被預測為原因子句
                                    cause_predictions[count_sentence] = (pred_token_id_val == 3221)  # 3221 是'是'的token_id
                                else:
                                    all_pass = False
                                    break
                            else:  # mask_type == 2, 第三個 [MASK] (配對預測)
                                # 檢查當前子句是否被預測為原因子句
                                if cause_predictions.get(count_sentence, False):
                                    # 如果被預測為原因子句，才預測配對的情緒子句編號
                                    case = [label_index[k] for k in range(
                                        max(0, -opt.window_size + count_sentence - 1),
                                        min(75, opt.window_size + count_sentence)
                                    )] # case: 合法的 token id list
                                    case.append(3187)  # '无'
                                    candidate_logits = logits[i, pos, case] # 取出pos位置在case對應的softmax機率
                                    selected_idx = torch.argmax(candidate_logits).item() # 在這些合法候選中，找出最大 softmax 機率的位置(索引)
                                    pred_token_id = case[selected_idx] # 假設pred_token_id = 125(代表配對到第4個子句)
                                    pseudo_label.append(pred_token_id if isinstance(pred_token_id, int) else pred_token_id.item())
                                else:
                                    # 如果不是原因子句，直接填入'无'
                                    pseudo_label.append(3187)  # 3187 是'无'的token_id
                        if all_pass:
                                # 這個樣本所有 [MASK] 都通過閾值，可以當 pseudo-labeled 樣本
                                pseudo_labeled_samples.append((batch_x[i].cpu().numpy(), np.array(pseudo_label)))
                                pseudo_labeled_indices.append(i)
                    print(f"本 batch 收集到 {len(pseudo_labeled_samples)} 筆 pseudo-labeled 樣本 (閥值: {threshold})")


                    # 收集pseudo預測結果用於最後一次性保存
                    pseudo_doc_ids = [unlabeled_dataset.doc_id[batch_start_idx + i] for i in pseudo_labeled_indices]
                    pseudo_x_bert = [x for x, y in pseudo_labeled_samples]
                    pseudo_labels = [y for x, y in pseudo_labeled_samples]
                    
                    # 將預測結果添加到all_pseudo_predictions
                    for i, doc_id in enumerate(pseudo_doc_ids):
                        all_pseudo_predictions.append({
                            'doc_id': doc_id,
                            'x_bert': pseudo_x_bert[i],
                            'pseudo_labels': pseudo_labels[i]
                        })

                    all_pseudo_labeled_samples.extend(pseudo_labeled_samples) # 將本 batch 的 pseudo-labeled 樣本與 doc_id 加入總表
                    all_pseudo_doc_ids.extend([unlabeled_dataset.doc_id[batch_start_idx + i] for i in pseudo_labeled_indices])

                    batch_start_idx += batch_x.shape[0]
                    
                    # 釋放當前batch的GPU記憶體和RAM
                    if use_gpu:
                        torch.cuda.empty_cache()
                    del logits  # 釋放當前batch logits
                        
                print(f"=== Self-training round {self_round+1} 結果 ===")
                print(f"總共收集到 {len(all_pseudo_labeled_samples)} 筆 pseudo-labeled 樣本")
                print(f"Unlabeled 資料集大小: {len(unlabeled_dataset)}")
                pseudo_ratio = len(all_pseudo_labeled_samples) / len(unlabeled_dataset) * 100 if len(unlabeled_dataset) > 0 else 0
                print(f"Pseudo-labeling 成功率: {pseudo_ratio:.2f}%")
                
                # 一次性保存所有pseudo預測結果
                if all_pseudo_predictions:
                    pseudo_x_bert_all = [item['x_bert'] for item in all_pseudo_predictions]
                    pseudo_labels_all = [item['pseudo_labels'] for item in all_pseudo_predictions]
                    pseudo_doc_ids_all = [item['doc_id'] for item in all_pseudo_predictions]
                    
                    save_mask_predictions(
                        logits=None,
                        x_bert=pseudo_x_bert_all,
                        tokenizer=tokenizer,
                        doc_ids=pseudo_doc_ids_all,
                        fold=fold,
                        output_dir="pseudo_results",
                        base_filename="pseudo_text_result",
                        pseudo_labels=pseudo_labels_all
                    )

            # 建立 pseudo-labeled dataset
            if len(all_pseudo_labeled_samples) > 0:
                pseudo_dataset = PseudoLabeledDataset(all_pseudo_labeled_samples, all_pseudo_doc_ids)
                print(f"  建立偽標籤資料集，共 {len(pseudo_dataset)} 筆樣本")
                
                # 合併 labeled 與 pseudo-labeled dataset
                combined_dataset = ConcatDataset([NLP_Dataset['train'], pseudo_dataset])
                print(f"  建立合併資料集: labeled={len(NLP_Dataset['train'])}, pseudo={len(pseudo_dataset)}")
            else:
                # 如果沒有pseudo-labeled樣本，只使用原始訓練資料
                combined_dataset = NLP_Dataset['train']
                print("  本輪pseudo-labeled樣本為0，只使用原始訓練資料")


            # 建立新的 DataLoader (包含初始10%有標籤樣本 + 當前第self_round的偽標籤樣本)
            combined_loader = DataLoader(combined_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)
            model.train()
            for batch_idx, batch in enumerate(combined_loader):
                x_bert, y_bert, label, mask_label, gt_emotion, gt_cause, gt_pair, is_labeled = batch
                labeled_mask = (is_labeled == True)
                pseudo_mask = (is_labeled == False)

                # 取出各自的資料
                if labeled_mask.sum() > 0:
                    x_l = x_bert[labeled_mask]
                    mask_label_l = mask_label[labeled_mask]
                    loss_l, _ = model(x_l.cuda() if use_gpu else x_l, mask_label_l.cuda() if use_gpu else mask_label_l)
                    loss_l = loss_l / labeled_mask.sum()
                else:
                    loss_l = 0

                if pseudo_mask.sum() > 0:
                    x_p = x_bert[pseudo_mask]
                    mask_label_p = mask_label[pseudo_mask]
                    loss_p, _ = model(x_p.cuda() if use_gpu else x_p, mask_label_p.cuda() if use_gpu else mask_label_p)
                    loss_p = loss_p / pseudo_mask.sum()
                else:
                    loss_p = 0

                total_loss = loss_l + opt.gamma * loss_p
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                


            # 在驗證集上測試當前模型性能（而不是測試集）
            model.eval()
            all_val_logits = torch.tensor([])
            all_val_label = torch.tensor([])
            all_val_mask_label = torch.tensor([])
            all_val_y_bert = torch.tensor([])
            all_val_x_bert = torch.tensor([])
            all_val_emotion_gt = torch.tensor([])
            all_val_cause_gt = torch.tensor([])
            all_val_pair_gt = torch.tensor([])

            with torch.no_grad():
                for _, data in enumerate(valloader):  # 改為驗證集
                    x_bert, y_bert, label, mask_label, gt_emotion, gt_cause, gt_pair, _ = data
                    if use_gpu:
                        x_bert = x_bert.cuda()
                        y_bert = y_bert.cuda()
                        label = label.cuda()
                        mask_label = mask_label.cuda()
                    loss, logits = model(x_bert, label)
                    logits = F.softmax(logits, dim=-1)
                    all_val_label = torch.cat((all_val_label, label.cpu()), 0)
                    all_val_mask_label = torch.cat((all_val_mask_label, mask_label.cpu()), 0)
                    all_val_logits = torch.cat((all_val_logits, logits.cpu()), 0)
                    all_val_y_bert = torch.cat((all_val_y_bert, y_bert.cpu()), 0)
                    all_val_x_bert = torch.cat((all_val_x_bert, x_bert.cpu()), 0)
                    all_val_emotion_gt = torch.cat((all_val_emotion_gt, gt_emotion), 0)
                    all_val_cause_gt = torch.cat((all_val_cause_gt, gt_cause), 0)
                    all_val_pair_gt = torch.cat((all_val_pair_gt, gt_pair), 0)

            p_emotion, r_emotion, f_emotion, p_cause, r_cause, f_cause, p_pair, r_pair, f_pair = crf_prompt(
                all_val_logits, all_val_label, all_val_x_bert, all_val_emotion_gt, all_val_cause_gt,
                all_val_pair_gt, save_path=os.path.join(st_results_dir, f"self_training_val_results_fold{fold}_round{self_round+1}.txt"))
            print(f"[Self-training round {self_round+1}] Validation: e_f: {f_emotion:.4f} c_f: {f_cause:.4f} pair_f: {f_pair:.4f}")
            
            # 初始化 self-training 的最佳性能追蹤變數
            if self_round == 0:
                st_max_f1_pair = -1.0
                st_max_p_pair = -1.0
                st_max_r_pair = -1.0
            
            # 檢查是否為當前最佳性能，並儲存模型
            if f_pair > st_max_f1_pair:
                st_max_f1_pair, st_max_p_pair, st_max_r_pair = f_pair, p_pair, r_pair
                if opt.savecheckpoint:
                    # 建立 self-training 專用的子資料夾
                    st_save_dir = os.path.join(save_path, 'self_training_models')
                    os.makedirs(st_save_dir, exist_ok=True)
                    st_model_path = os.path.join(st_save_dir, f'fold{fold}_self_training_best.pth')
                    torch.save(model, st_model_path)
                    print(f"   Self-training 最佳模型已儲存: {st_model_path}")
                    print(f"   當前最佳 pair F1: {st_max_f1_pair:.4f}")
            
            import psutil
            memory_info = psutil.virtual_memory()
            current_time = time.strftime("%H:%M:%S", time.localtime())
            round_duration = time.time() - round_start_time
            print(f"記憶體使用: {memory_info.percent}% ({memory_info.used // 1024**3}GB/{memory_info.total // 1024**3}GB) - 時間: {current_time}")
            print(f"Round {self_round+1} 執行時間: {round_duration/60:.1f} 分鐘")
            
        
        # Self-training 結束，計算總時間
        self_training_total_time = time.time() - self_training_start_time
        print(f"\nSelf-training 完成！")
        print(f"Self-training 總執行時間: {self_training_total_time/60:.1f} 分鐘")
        
        # 當前 fold 結束，計算總時間
        fold_total_time = time.time() - fold_start_time
        print(f"\nFold {fold} 完成！")
        print(f"Fold {fold} 總執行時間: {fold_total_time/60:.1f} 分鐘")
        print(f"Fold {fold} 結束時間: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        
        # 記錄當前fold的時間信息
        fold_time_info = {
            'fold': fold,
            'total_time': fold_total_time,
            'supervised_time': supervised_training_total_time,
            'self_training_time': self_training_total_time,
            'is_test_only': False  # 標記為訓練模式
        }
        fold_times.append(fold_time_info)
        
        # 即時寫入當前fold的時間記錄
        with open(time_log_file, "a", encoding="utf-8") as f:
            f.write(f"Fold {fold} 時間記錄:\n")
            f.write(f"  開始時間: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(fold_start_time))}\n")
            f.write(f"  結束時間: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
            f.write(f"  總執行時間: {fold_total_time/60:.1f} 分鐘\n")
            f.write(f"  初始監督訓練時間: {supervised_training_total_time/60:.1f} 分鐘\n")
            f.write(f"  Self-training 時間: {self_training_total_time/60:.1f} 分鐘\n")
            f.write("-" * 30 + "\n\n")
    
    # 所有 fold 結束，計算總時間
    all_folds_total_time = time.time() - all_folds_start_time
    overall_total_time = time.time() - overall_start_time
    print(f"\n所有實驗完成！")
    print(f"所有 fold 總執行時間: {all_folds_total_time/60:.1f} 分鐘")
    print(f"程式總執行時間: {overall_total_time/60:.1f} 分鐘")
    print(f"實驗結束時間: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    
    # 寫入最終總結時間記錄
    with open(time_log_file, "a", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write("=== 總結時間統計 ===\n")
        f.write(f"實驗結束時間: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
        f.write(f"所有 fold 總執行時間: {all_folds_total_time/60:.1f} 分鐘\n")
        f.write(f"程式總執行時間: {overall_total_time/60:.1f} 分鐘\n\n")
        
        # 檢查是否為test_only模式
        is_test_only_mode = any(fold_info.get('is_test_only', False) for fold_info in fold_times)
        
        if is_test_only_mode:
            # 測試模式
            f.write("=== 測試模式時間統計 ===\n")
            f.write("注意：此次執行為純測試模式 (test_only=True)，未進行任何訓練\n\n")
        else:
            # 訓練模式
            f.write("=== 訓練模式時間統計 ===\n")
        
        # 詳細統計每個fold
        f.write("=== 各 Fold 時間詳細統計 ===\n")
        total_supervised_time = 0
        total_self_training_time = 0
        
        for fold_info in fold_times:
            total_supervised_time += fold_info['supervised_time']
            total_self_training_time += fold_info['self_training_time']
            
            # 根據模式顯示不同的時間信息
            if fold_info.get('is_test_only', False):
                f.write(f"Fold {fold_info['fold']}: {fold_info['total_time']/60:.1f}分 (純測試時間)\n")
            else:
                f.write(f"Fold {fold_info['fold']}: {fold_info['total_time']/60:.1f}分 "
                       f"(監督: {fold_info['supervised_time']/60:.1f}分, "
                       f"Self-training: {fold_info['self_training_time']/60:.1f}分)\n")
        
        f.write(f"\n平均每折時間: {all_folds_total_time/len(fold_times)/60:.1f} 分鐘\n" if len(fold_times) > 0 else "\n平均每折時間: 無資料\n")
        
        # 根據模式顯示不同的總計信息
        if is_test_only_mode:
            f.write("總測試時間: {:.1f} 分鐘\n".format(sum(fold_info['total_time'] for fold_info in fold_times)/60))
        else:
            f.write(f"總監督訓練時間: {total_supervised_time/60:.1f} 分鐘\n")
            f.write(f"總 Self-training 時間: {total_self_training_time/60:.1f} 分鐘\n")
        
        f.write("=" * 50 + "\n")
    
    print(f"時間記錄已保存至: {time_log_file}")




if __name__ == '__main__':
    run()