import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import AutoTokenizer, default_data_collator

max_seq_length = 512
doc_stride = 128
tokenizer_path = "./whatisit"
train_file = './data/VU_squad2.0_train.json'
batch_size = 16


def get_tokenizer(tokenizer_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=True
    )
    return tokenizer


def get_raw_datasets(train_file: str):
    data_files = {}
    data_files["train"] = train_file
    extension = train_file.split(".")[-1]

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        field="data",
    )
    return raw_datasets


def get_datasets():
    tokenizer = get_tokenizer(tokenizer_path=tokenizer_path)
    raw_datasets = get_raw_datasets(train_file=train_file)

    column_names = raw_datasets["train"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    # Training preprocessing
    def prepare_train_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    train_dataset = raw_datasets["train"]
    train_dataset = train_dataset.map(
        prepare_train_features,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on train dataset",
    )
    return train_dataset


# class myDatasets(Dataset):
#     def __init__(self, train_dataset):
#         self.train_dataset = train_dataset
#
#     def __getitem__(self, index):
#         return {
#             "input_ids": torch.Tensor(self.train_dataset["input_ids"][index]).to(torch.long),
#             "attention_mask": torch.Tensor(self.train_dataset["attention_mask"][index]).to(torch.long),
#             "token_type_ids": torch.Tensor(self.train_dataset["token_type_ids"][index]).to(torch.long),
#             "start_positions": self.train_dataset["start_positions"][index],
#             "end_positions": self.train_dataset["end_positions"][index]
#         }
#
#     def __len__(self):
#         return len(self.train_dataset)


def get_train_dataloader(batch_size: int) -> DataLoader:
    train_dataset = get_datasets()
    data_collator = default_data_collator

    train_sampler = RandomSampler(train_dataset)

    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=data_collator,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
        # worker_init_fn=seed_worker,
    )


if __name__ == '__main__':
    train_dataloader = get_train_dataloader(batch_size=batch_size)

    print(next(iter(train_dataloader)))
