import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModel,
    AutoTokenizer,
    BertForQuestionAnswering,
    BertTokenizer,
)

from data_modules import (
    ContrastiveIntentDataModule,
    IntentDataModule,
    SlotNameDataModule,
)
from models import ContrastiveIntentModel, SlotNameModel


class Evaluate:
    def __init__(self, intent_path="", slot_name_path="") -> None:
        #    self.intent_model =  SentenceTransformer.from_pretrained(intent_path)
        self.slot_name_model = SlotNameModel.load_from_checkpoint(slot_name_path)
        self.slot_name_model.eval()
        self.intent_model = ContrastiveIntentModel.load_from_checkpoint(intent_path)
        self.intent_model.eval()

        self.qa_tokenizer = BertTokenizer.from_pretrained(
            "healx/biomedical-slot-filling-reader-base"
        )
        self.qa_model = BertForQuestionAnswering.from_pretrained(
            "healx/biomedical-slot-filling-reader-base"
        )

    def test(self):

        step = "test"
        # i_dm = ContrastiveIntentDataModule(batch_size=1)
        # i_dm.setup()
        # count = 0
        # all_intents = np.array(self.intent_model.all_intents[step])
        # utt, i_preds = [], []
        # for utterances, intents, intents_str in i_dm.train_dataloader():
        #     self.intent_model.model = self.intent_model.model.to(
        #         self.intent_model.device
        #     )
        #     i_loss = self.intent_model.criterion([utterances, intents], 1)
        #     similarities = self.intent_model(utterances, all_intents, test=True)
        #     val, indx = torch.topk(similarities, 1, dim=1)
        #     intent_preds = str(all_intents[indx])
        #     utterances_text = i_dm.tokenizer.batch_decode(
        #         utterances["input_ids"], skip_special_tokens=True
        #     )
        #     utt.append(utterances_text[0])
        #     i_preds.append(intent_preds)
        #     count += 1
        #     if count == 10:
        #         break

        # with open("eval_intents.txt", "w") as f:
        #     for u, i in zip(utt, i_preds):
        #         f.write("-" * 10)
        #         f.write(f"\nutterance : {u}\n")
        #         f.write(f"\nintent pred: {i}\n")

        dm = SlotNameDataModule(batch_size=1)
        dm.setup()

        count = 0
        all_slot_names = np.array(self.slot_name_model.slot_names[step])
        utt_out, slot_pred_out, ans_out = [], [], []
        for utterances, slots, labels in dm.train_dataloader():
            loss = self.slot_name_model.criterion([utterances, slots], 1)
            similarities = self.slot_name_model(utterances, all_slot_names, test=True)
            val, indx = torch.topk(similarities, 1, dim=1)
            slots_preds = str(all_slot_names[indx])
            utterances_text = dm.tokenizer.batch_decode(
                utterances["input_ids"], skip_special_tokens=True
            )
            utt = utterances_text[0]
            q_inputs = self.qa_tokenizer(utt, slots_preds, return_tensors="pt")
            outputs = self.qa_model(**q_inputs)
            answer_start_index = outputs.start_logits[:, 1:].argmax()
            answer_end_index = outputs.end_logits[:, 1:].argmax()
            predict_answer_tokens = q_inputs.input_ids[
                0, answer_start_index : answer_end_index + 1
            ]
            ans = self.qa_tokenizer.decode(predict_answer_tokens)
            utt_out.append(utt)
            slot_pred_out.append(slots_preds)
            ans_out.append(ans)

            # intent detection
            count += 1
            if count == 50:
                break
            a = 1

        with open("eval_slot_name.txt", "w") as f:
            for u, s, a in zip(utt_out, slot_pred_out, ans_out):
                f.write("-" * 10)
                f.write(f"\nutterance : {u}\n")
                f.write(f"\nslot name prediction : {s}\n")
                f.write(f"\nSlots Values: {a}\n")


if __name__ == "__main__":
    s_path = "lightning_logs/slot_name/lightning_logs/version_0/checkpoints/epoch=0-step=30.ckpt"
    i_path = "intent/lightning_logs/version_6/checkpoints/epoch=49-step=35199.ckpt"
    e = Evaluate(slot_name_path=s_path, intent_path=i_path)
    e.test()
