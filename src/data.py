
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

        def make_chat(inp):
            chat = ["[Conversation]"]
            keyword_sentence_id = inp['issue']['sentence_id']

            # keyword_sentence 근처 40개의 대화만 포함
            if len(inp['conversation']) > 40:
                start_id = int(keyword_sentence_id.split('.')[-1]) - 20
                if start_id < 0:
                    start_id = 0

                end_id = start_id + 40
                if end_id > len(inp['conversation']):
                    end_id = len(inp['conversation'])
                    start_id = end_id - 40
            else:
                start_id = 0
                end_id = len(inp['conversation'])

            for cvt in inp['conversation'][start_id:end_id]:
                speaker = cvt['speaker']
                utterance = cvt['utterance']
                chat.append(f"화자 {speaker}: {utterance}")
            chat = "\n".join(chat)

            question = f"[Question]\n위 대화에서 토픽 {', '.join(inp['issue']['topic'])}에 대한 대화를 요약해주세요."
            chat = chat + "\n\n" + question

            return chat
        
        for example in data:
            chat = make_chat(example["input"])
            message = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": chat},
            ]
     
            source = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            target = example["output"]
            if target != "":
                target += tokenizer.eos_token
            target = tokenizer(target,
                      return_attention_mask=False,
                      add_special_tokens=False,
                      return_tensors="pt",
                      max_length=1024, truncation=True)
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
