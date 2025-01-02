from .tokenizer import tokenizer
from .enums import EmotionType
from .model import EmotionClassifier
from .dataset import InferenceDataset
import torch


use_cuda = torch.cuda.is_available()
if use_cuda:
    print("CUDA is available! Using it...")
else:
    print("CUDA isn't available, using CPU...")
device = torch.device("cuda" if use_cuda else "cpu")


def loadModel():
    model = EmotionClassifier()
    model.to(device)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    return model


def classifyTextBatch(model, *texts):
    data_loader = torch.utils.data.DataLoader(InferenceDataset(texts),
                                              batch_size=1)
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            return [EmotionType(int(pred)) for pred in preds]


def classifyText(model, text):
    tokens = tokenizer(text,
                       padding='max_length', max_length=512, truncation=True,
                       return_tensors="pt")

    mask = tokens['attention_mask'].to(device)
    input_id = tokens['input_ids'].to(device)

    with torch.no_grad():
        outputs = model(input_id, mask)
        _, preds = torch.max(outputs, dim=1)
    return EmotionType(int(preds))
