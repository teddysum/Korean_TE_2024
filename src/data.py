
import json

import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, fname, tokenizer):
        IGNORE_INDEX=-100
        self.inp = []
        self.label = []

        PROMPT = '''You are a helpful AI assistant. Please answer the user's questions kindly. 당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요.'''

        with open(fname, "r") as f:
            data = json.load(f)

        def make_prompt_input(inp):
            input_prompt = "다음은 함의분석 결과입니다. premise와 proposition의 관계에 대해 entailment 또는 contradiction 로 주석 되어 있습니다.\n"

            premise = "premise: " + inp['premise']
            input_prompt += premise + '\n'

            proposition = "premise: " + inp['proposition']
            input_prompt += proposition + '\n'

            label = "label: " + inp['label']
            input_prompt += label + '\n'

            question = f"[Question]\n위 함의 분석 결과에서 premise와 proposition의 관계가 왜 label인지 설명문을 생성해 주세요."
            final_input_prompt = input_prompt + "\n\n" + question

            return final_input_prompt
        
        for example in data:
            chat = make_prompt_input(example["input"])
            message = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": chat},
            ]
     
            source = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            if 'output' not in example:
                example["output"] = ""

            target = example["output"]
            if target != "":
                target += tokenizer.eos_token

            target = tokenizer(target,
                      return_attention_mask=False,
                      add_special_tokens=False,
                      return_tensors="pt")
            target["input_ids"] = target["input_ids"].type(torch.int64)

            input_ids = torch.concat((source[0], target["input_ids"][0]))
            labels = torch.concat((torch.LongTensor([IGNORE_INDEX] * source[0].shape[0]), target["input_ids"][0]))
            self.inp.append(input_ids)
            self.label.append(labels)

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return self.inp[idx]


class DataCollatorForSupervisedDataset(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids) for ids in input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(lbls) for lbls in labels], batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
