import os
import args
import torch
import random
import pickle
from tqdm import tqdm
from torch import nn, optim
from collections import OrderedDict
import evaluate
from optimizer import BertAdam
from dataset.dataloader import Dureader
from dataset.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from model_dir.modeling import BertForQuestionAnswering, BertConfig


class FGM(object):
    def __init__(self, model):
        super(FGM, self).__init__()
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.25, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# 随机种子
random.seed(args.seed)
torch.manual_seed(args.seed)
device = args.device
device_ids = [0]
if len(device_ids) > 0:
    torch.cuda.manual_seed_all(args.seed)

def train():
    # s加载预训练bert
    model = BertForQuestionAnswering.from_pretrained('./roberta_wwm_ext',
                    cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(-1)))
    # output_model_file = "./model_dir_/baseextra"
    # output_config_file = "./model_dir/bert_config.json"
    #
    # config = BertConfig(output_config_file)
    # model = BertForQuestionAnswering(config)
    # # 针对多卡训练加载模型的方法：
    # state_dict = torch.load(output_model_file, map_location='cuda:0')
    # # 初始化一个空 dict
    # new_state_dict = OrderedDict()
    # # 修改 key，没有module字段则需要不上，如果有，则需要修改为 module.features
    # for k, v in state_dict.items():
    #     if 'module' not in k:
    #         k = k
    #     else:
    #         k = k.replace('module.', '')
    #     new_state_dict[k] = v
    # model.load_state_dict(new_state_dict)

    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)  # 声明所有可用设备
        model = model.cuda(device=device_ids[0])  # 模型放在主设备
    elif len(device_ids) == 1:
        model = model.cuda()

    # 准备 optimizer
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=0.1,
                         t_total=args.num_train_optimization_steps)
    # optimizer = nn.DataParallel(optimizer, device_ids=device_ids)
    # 准备数据
    data = Dureader()
    train_dataloader, dev_dataloader = data.train_iter, data.dev_iter

    best_loss = 100000.0
    model.train()
    # 初始化
    fgm = FGM(model)
    for i in range(args.num_train_epochs):
        for step, batch in enumerate(tqdm(train_dataloader, desc="Epoch")):
            input_ids, input_mask, segment_ids, start_positions, end_positions = \
                batch.input_ids, batch.input_mask, batch.segment_ids, batch.start_position, batch.end_position
            if len(device_ids) > 1:
                input_ids, input_mask, segment_ids, start_positions, end_positions = \
                    input_ids.cuda(device=device_ids[0]), input_mask.cuda(device=device_ids[0]), \
                    segment_ids.cuda(device=device_ids[0]), start_positions.cuda(device=device_ids[0]), \
                    end_positions.cuda(device=device_ids[0])
            elif len(device_ids) == 1:
                input_ids, input_mask, segment_ids, start_positions, end_positions = \
                    input_ids.cuda(), input_mask.cuda(), \
                    segment_ids.cuda(), start_positions.cuda(), \
                    end_positions.cuda()
            # 正常训练
            # 计算loss
            loss, s, e = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                               start_positions=start_positions, end_positions=end_positions)

            # loss = loss / args.gradient_accumulation_steps
            # loss.backward()
            # if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()

            # 对抗训练
            fgm.attack()  # 在embedding上添加对抗扰动
            loss_adv, s, e = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                                   start_positions=start_positions, end_positions=end_positions)
            loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            fgm.restore()  # 恢复embedding参数
            # 更新梯度
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # 验证
            if step % args.log_step == 4:
                # torch.save(model.state_dict(), './model_dir/' + "best_model_baseextraadv_d")
                eval_loss = evaluate.evaluate(model, dev_dataloader, device_ids)
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    if len(device_ids) > 1:
                        torch.save(model.module.state_dict(), './model_dir/' + "best_model_base")
                    if len(device_ids) == 1:
                        torch.save(model.state_dict(), './model_dir/' + "best_model_base")
                    model.train()


if __name__ == "__main__":
    train()

