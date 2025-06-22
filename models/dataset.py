from torch.utils.data import Dataset
from collections import defaultdict
import torch

class nerDataset(Dataset):
    def __init__(self, data, word2idx=None, tag2idx=None):
        self.sentences = []
        self.labels = []
        
        current_sentence = []
        current_tags = []

        for row in data:
            if len(row) == 0 or row[0].strip() == "":
                if current_sentence:
                    self.sentences.append(current_sentence)
                    self.labels.append(current_tags)
                    current_sentence, current_tags = [], []
            elif len(row) >=2 :
                word = row[0]
                tag = row[-1]
                current_sentence.append(word)
                current_tags.append(tag)
            else: 
                continue

        if current_sentence:
            self.sentences.append(current_sentence)
            self.labels.append(current_tags)

        if word2idx is None:
            words = set(w for sent in self.sentences for w in sent)
            self.word2idx = {w: i + 2 for i, w in enumerate(words)}
            self.word2idx["<PAD>"] = 0
            self.word2idx["<UNK>"] = 1
        else:
            self.word2idx = word2idx

        if tag2idx is None:
            tags = set(t for tags in self.labels for t in tags)
            self.tag2idx = {t: i for i, t in enumerate(sorted(tags))}
        else:
            self.tag2idx = tag2idx

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sent = self.sentences[idx]
        tags = self.labels[idx]
        word_ids = [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in sent]
        tag_ids = [self.tag2idx[t] for t in tags]
        return word_ids, tag_ids


def collate_fn(batch):
    max_len = max(len(x[0]) for x in batch)
    
    padded_words = []
    padded_tags = []

    for word_ids, tag_ids in batch:
        pad_len = max_len - len(word_ids)
        
        padded_words.append(word_ids + [0] * pad_len)  
        padded_tags.append(tag_ids + [0] * pad_len)    

    return torch.tensor(padded_words, dtype=torch.long), torch.tensor(padded_tags, dtype=torch.long)

    


from torch.utils.data import DataLoader
from pathlib import Path
ROOT_DIR = Path.cwd()

data_path = ROOT_DIR / "data" / "bio_output.txt"

with open(data_path, "r", encoding="utf-8") as f:
    lines = [line.strip().split() for line in f.readlines()]

print("Dd")
dataset = nerDataset(lines)

loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

for X_batch, y_batch in loader:
    print("Words:", X_batch.shape)
    print("Tags :", y_batch.shape)
    break

