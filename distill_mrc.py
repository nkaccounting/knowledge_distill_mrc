from textbrewer import GeneralDistiller
from textbrewer import TrainingConfig, DistillationConfig
from transformers import AutoModelForQuestionAnswering
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertConfig

from data_loader import get_train_dataloader


def main():
    print("load dataloader---------------------------")
    train_dataloader = get_train_dataloader(batch_size=4)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    teacher_model = AutoModelForQuestionAnswering.from_pretrained("./whatisit")

    bert_config_T3 = BertConfig.from_json_file(
        './student_config/bert_base_cased_config/bert_config_L3.json')

    bert_config_T3.output_hidden_states = True
    teacher_model.config.output_hidden_states = True

    student_model = AutoModelForQuestionAnswering.from_config(bert_config_T3)

    teacher_model.to(device=device)
    student_model.to(device=device)

    num_epochs = 4
    num_training_steps = len(train_dataloader) * num_epochs
    # Optimizer and learning rate scheduler
    optimizer = AdamW(student_model.parameters(), lr=1e-4)

    scheduler_class = get_linear_schedule_with_warmup
    # arguments dict except 'optimizer'
    scheduler_args = {'num_warmup_steps': int(0.1 * num_training_steps), 'num_training_steps': num_training_steps}

    def simple_adaptor(batch, model_outputs):
        return {
            'logits': (model_outputs.start_logits, model_outputs.end_logits),
            'hidden': model_outputs.hidden_states,
            'attention': model_outputs.attentions
        }

    distill_config = DistillationConfig(
        temperature=10,
        hard_label_weight=0.5,
        kd_loss_weight=0.5,
        intermediate_matches=[{"layer_T": [0, 0], "layer_S": [0, 0], "feature": "hidden", "loss": "mmd", "weight": 1},
                              {"layer_T": [8, 8], "layer_S": [1, 1], "feature": "hidden", "loss": "mmd", "weight": 1},
                              {"layer_T": [16, 16], "layer_S": [2, 2], "feature": "hidden", "loss": "mmd", "weight": 1},
                              {"layer_T": [24, 24], "layer_S": [3, 3], "feature": "hidden", "loss": "mmd",
                               "weight": 1}])

    train_config = TrainingConfig(
        device=device,
        log_dir='./distill_log',
        output_dir='./distill_path',
        ckpt_frequency=2,
        ckpt_epoch_frequency=1,
    )

    distiller = GeneralDistiller(
        train_config=train_config,
        distill_config=distill_config,
        model_T=teacher_model,
        model_S=student_model,
        adaptor_T=simple_adaptor,
        adaptor_S=simple_adaptor
    )

    step = 1

    def batch_postprocessor(batch):
        nonlocal step
        print(step)
        step += 1
        res = {"input_ids": batch['input_ids'],
               "attention_mask": batch['attention_mask'],
               "token_type_ids": batch['token_type_ids'],
               "start_positions": batch['start_positions'],
               "end_positions": batch['end_positions']}
        return res

    print("distiller---------------------------")
    with distiller:
        distiller.train(optimizer, train_dataloader, num_epochs, scheduler_class=scheduler_class,
                        scheduler_args=scheduler_args, batch_postprocessor=batch_postprocessor, callback=None)


if __name__ == '__main__':
    main()
