import argparse
import os
import sys
import torch.nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM
import time
import numpy as np
import json
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
parser.add_argument('--training_iter', type=int, default=3, help='number of train iterator')
parser.add_argument('--scope', type=str, default='Ind_BiLSTM', help='scope')
parser.add_argument('--batch_size', type=int, default=8, help='number of example per batch')
parser.add_argument('--learning_rate', type=float, default=0.00001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay for bert')
parser.add_argument('--usegpu', type=bool, default=True, help='gpu')
"""other"""
parser.add_argument('--test_only', type=bool, default=False, help='no training')
parser.add_argument('--checkpoint', type=bool, default=False, help='load checkpoint')
parser.add_argument('--checkpointpath', type=str, default='checkpoint/ECPE/', help='path to load checkpoint')
parser.add_argument('--savecheckpoint', type=bool, default=True, help='save checkpoint')
parser.add_argument('--save_path', type=str, default='prompt_ECPE_few_shot_ST', help='path to save checkpoint')
parser.add_argument('--device', type=str, default='0', help='device id')
parser.add_argument('--dataset', type=str, default='split10_few_shot_ST/', help='path for dataset')
parser.add_argument('--start_fold', type=int, default=1)
parser.add_argument('--end_fold',   type=int, default=3)
"""self-training parameters"""
parser.add_argument('--threshold', type=float, default=0.9, help='confidence threshold for pseudo-labeling')
parser.add_argument('--gamma', type=float, default=1.0, help='weight for pseudo-labeled loss')
parser.add_argument('--lambda_reg', type=float, default=1e-4, help='L2 regularization weight')

opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.device

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
                     self.gt_emotion[index], self.gt_cause[index], self.gt_pair[index]]
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
                else:
                    # count_mask == 0 → cause
                    if count_mask == 0 or count_mask == 1:
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

    print_time()
    overall_start_time = time.time()
    bert_path = './bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(bert_path)

    # train
    print_training_info()  # 輸出訓練的超參數資訊

    max_result_emo_f, max_result_emo_p, max_result_emo_r = [], [], []
    max_result_pair_f, max_result_pair_p, max_result_pair_r = [], [], []
    max_result_cause_f, max_result_cause_p, max_result_cause_r = [], [], []
    for fold in range(opt.start_fold, opt.end_fold + 1):
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
        test_file_name = 'fold{}_test.json'.format(fold)
        unlabeled_file_name = 'fold{}_unlabeled.json'.format(fold)
        unlabeled = opt.dataset + unlabeled_file_name


        print('############# fold {} begin ###############'.format(fold))
        train = opt.dataset + train_file_name
        test = opt.dataset + test_file_name
        edict = {"train": train, "test": test}
        unlabeled_dataset = UnlabeledDataset(unlabeled, tokenizer=tokenizer)
        NLP_Dataset = {x: MyDataset(edict[x], test=(x == 'test'), tokenizer=tokenizer) for x in ['train', 'test']}
        trainloader = DataLoader(NLP_Dataset['train'], batch_size=opt.batch_size, shuffle=True, drop_last=True)
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
        if opt.test_only:
            all_test_logits = torch.tensor([])
            all_test_label = torch.tensor([])
            all_test_mask_label = torch.tensor([])
            all_test_y_bert = torch.tensor([])
            all_test_x_bert = torch.tensor([])
            all_test_emotion_gt = torch.tensor([])
            all_test_cause_gt = torch.tensor([])
            all_test_pair_gt = torch.tensor([])
            model.eval()
            output_file_path = os.path.join(opt.save_path, 'test_results.txt')
            with torch.no_grad(), open(output_file_path, 'w', encoding='utf-8') as output_file:
                for idx, data in enumerate(testloader):
                    x_bert, y_bert, label, mask_label, gt_emotion, gt_cause, gt_pair = data
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
                    all_test_pair_gt, save_path="evaluation_results_few_shot_st.txt")
                save_mask_predictions(all_test_logits, all_test_x_bert, tokenizer, NLP_Dataset['test'].doc_id, fold=fold)
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

        else:
            for i in range(opt.training_iter):
                model.train()
                start_time, step = time.time(), 1
                # 此迴圈根據訓練資料集大小決定要跑幾次: 假設175筆資料、batch_size=8，\
                # 則此迴圈會跑21次(175/8=21.875，drop_last=True會捨去最後不足8筆資料) \ 
                # 21 * 8 = 168筆資料，175-168=7筆資料，所以會有7筆資料沒有被訓練到
                for index, data in enumerate(trainloader):
                    with torch.autograd.set_detect_anomaly(True):
                        x_bert, y_bert, label, mask_label, gt_emotion, gt_cause, gt_pair = data
                        if use_gpu:
                            x_bert = x_bert.cuda()
                            y_bert = y_bert.cuda()
                            label = label.cuda()
                            mask_label = mask_label.cuda()



                        loss, logits = model(x_bert, mask_label)
                        #if index == 0:
                        #    print("x_bert[0]:", x_bert[0])
                        #    print("x_bert[0] decoded:", tokenizer.decode(x_bert[0].cpu().tolist()))
                        #    print("mask_label[0]:", mask_label[0])
                        #    print("mask_label[0] decoded:", tokenizer.decode(mask_label[0].cpu().tolist()))
                        #    print("label[0]:", label[0])
                        #    print("label[0] decoded:", tokenizer.decode(label[0].cpu().tolist()))
                        #    print("label[0] == mask_label[0]:", (label[0] == mask_label[0]))
                        #    print("label[0] where != mask_label[0]:", torch.where(label[0] != mask_label[0]))
                        #    pred_token_ids = torch.argmax(logits[0], dim=-1)  # shape: [seq_len]
                        #    print("預測 token ids:", pred_token_ids)
                        #    print("預測文字:", tokenizer.decode(pred_token_ids.cpu().tolist()))
                        #    sys.exit()
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
                                crf_prompt(logits.cpu(), label.cpu(), x_bert.cpu(), gt_emotion, gt_cause, gt_pair, save_path="evaluation_results_few_shot_st.txt")
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
                all_test_logits = torch.tensor([])
                all_test_label = torch.tensor([])
                all_test_mask_label = torch.tensor([])
                all_test_y_bert = torch.tensor([])
                all_test_x_bert = torch.tensor([])
                all_test_emotion_gt = torch.tensor([])
                all_test_cause_gt = torch.tensor([])
                all_test_pair_gt = torch.tensor([])

                model.eval()
                with torch.no_grad():
                    for _, data in enumerate(testloader):
                        x_bert, y_bert, label, mask_label, gt_emotion, gt_cause, gt_pair = data
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
                        all_test_pair_gt, save_path="evaluation_results_few_shot_st.txt")
                    print("iter{} test result:".format(i))
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
                        if opt.savecheckpoint:
                            torch.save(model, save_path + '/' + 'fold{}.pth'.format(fold))
                            print(f"Model for fold {fold} saved to {save_path}/fold{fold}.pth")
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

            # ====== 當前第 i 折最佳模型的路徑 ======
            model_path = save_path + '/' + 'fold{}.pth'.format(fold)
            model = torch.load(model_path, map_location=torch.device('cuda' if use_gpu else 'cpu'))
            if use_gpu:
                model = model.cuda()
            print(f"已載入初始化監督訓練的最佳模型: {model_path}")

            

            print("=== 開始 Self-Training 流程 ===")
            os.makedirs("pseudo_results", exist_ok=True)
            while True:
                with torch.no_grad():
                    model.eval()
                    all_unlabeled_logits = []
                    all_unlabeled_x = []
                    batch_start_idx = 0
                    for batch_x in unlabeled_loader:
                        if use_gpu:
                            batch_x = batch_x.cuda()
                        _, logits = model(batch_x, labels=None)
                        logits = F.softmax(logits, dim=-1)
                        all_unlabeled_logits.append(logits.cpu())
                        all_unlabeled_x.append(batch_x.cpu())
                        print("第一個 batch 預測完成，logits shape:", logits.shape)

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
                            for j, pos in enumerate(mask_positions): # 這個 for 迴圈會跑 len(mask_positions) 輪，也就是 N × 3 輪（N = 子句數)/
                                # j: 目前是第幾個 [MASK]
                                # pos: 這個 [MASK] 在 token 序列中的實際索引位置
                                if j % 3 == 2:  # 第三個 [MASK]，只在合法候選中選最大 (0:情緒, 1:原因, 2:配對)
                                    count_sentence = j // 3 + 1 # 這是第幾個子句
                                    case = [label_index[k] for k in range(
                                        max(0, -opt.window_size + count_sentence - 1),
                                        min(75, opt.window_size + count_sentence)
                                    )] # case: 合法的 token id list
                                    case.append(3187)  # '无'
                                    candidate_logits = logits[i, pos, case] # 取出pos位置在case對應的softmax機率
                                    selected_idx = torch.argmax(candidate_logits).item() # 在這些合法候選中，找出最大 softmax 機率的位置(索引)
                                    pred_token_id = case[selected_idx] # 假設pred_token_id = 125(代表配對到第4個子句)
                                    prob = candidate_logits[selected_idx]
                                else:  # 不是第3個[MASK] 則直接取最大值填入
                                    prob, pred_token_id = torch.max(logits[i, pos], dim=-1)
                                if prob.item() >= threshold:
                                    pseudo_label.append(pred_token_id if isinstance(pred_token_id, int) else pred_token_id.item())
                                else:
                                    all_pass = False
                                    break
                            if all_pass:
                                # 這個樣本所有 [MASK] 都通過閾值，可以當 pseudo-labeled 樣本
                                pseudo_labeled_samples.append((batch_x[i].cpu().numpy(), np.array(pseudo_label)))
                                pseudo_labeled_indices.append(i)
                        print(f"本 batch 收集到 {len(pseudo_labeled_samples)} 筆 pseudo-labeled 樣本")

                        pseudo_doc_ids = [unlabeled_dataset.doc_id[batch_start_idx + i] for i in pseudo_labeled_indices]
                        pseudo_x_bert = [x for x, y in pseudo_labeled_samples]
                        pseudo_labels = [y for x, y in pseudo_labeled_samples]

                        save_mask_predictions(
                            logits=None,
                            x_bert=pseudo_x_bert,
                            tokenizer=tokenizer,
                            doc_ids=pseudo_doc_ids,
                            fold=fold,
                            output_dir="pseudo_results",
                            base_filename="pseudo_text_result",
                            pseudo_labels=pseudo_labels
                        )
                        batch_start_idx += batch_x.shape[0]
                        sys.exit(0)

                        # 你可以在這裡進一步處理 pseudo_labeled_samples
                    all_unlabeled_logits = torch.cat(all_unlabeled_logits, dim=0)
                    all_unlabeled_x = torch.cat(all_unlabeled_x, dim=0)
                print("unlabeled 預測完成，logits shape:", all_unlabeled_logits.shape)


    
    # print("emotion")
    # print(max_result_emo_f)
    # print("average f {:.4f}".format(sum(max_result_emo_f) / len(max_result_emo_f)))
    # print(max_result_emo_p)
    # print("average p {:.4f}".format(sum(max_result_emo_p) / len(max_result_emo_p)))
    # print(max_result_emo_r)
    # print("average r {:.4f}".format(sum(max_result_emo_r) / len(max_result_emo_r)))
    # print("cause")
    # print(max_result_cause_f)
    # print("average f {:.4f}".format(sum(max_result_cause_f) / len(max_result_cause_f)))
    # print(max_result_cause_p)
    # print("average p {:.4f}".format(sum(max_result_cause_p) / len(max_result_cause_p)))
    # print(max_result_cause_r)
    # print("average r {:.4f}".format(sum(max_result_cause_r) / len(max_result_cause_r)))
    # print("pair")
    # print(max_result_pair_f)
    # print("average f {:.4f}".format(sum(max_result_pair_f) / len(max_result_pair_f)))
    # print(max_result_pair_p)
    # print("average p {:.4f}".format(sum(max_result_pair_p) / len(max_result_pair_p)))
    # print(max_result_pair_r)
    # print("average r {:.4f}".format(sum(max_result_pair_r) / len(max_result_pair_r)))
    # overall_time = time.time() - overall_start_time
    # print(f"\nTotal training time: {overall_time // 3600:.0f} hours, "
    #      f"{(overall_time % 3600) // 60:.0f} minutes, and {overall_time % 60:.2f} seconds.")

if __name__ == '__main__':
    run()