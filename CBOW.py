import torch
import torch.nn as nn
text = """People who truly loved once are far more likely to love again.
Difficult circumstances serve as a textbook of life for people.
The best preparation for tomorrow is doing your best today.
The reason why a great man is great is that he resolves to be a great man.
The shortest way to do many things is to only one thing at a time.
Only they who fulfill their duties in everyday matters will fulfill them on great occasions. 
I go all out to deal with the ordinary life. 
I can stand up once again on my own.
Never underestimate your power to change yourself.""".split()

word = set(text)
word_size = len(word)

word_to_ix = {word:ix for ix, word in enumerate(word)}
ix_to_word = {ix:word for ix, word in enumerate(word)}

def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)

EMDEDDING_DIM = 100

data = []
for i in range(2, len(text) - 2):
    context = [text[i - 2], text[i - 1],
               text[i + 1], text[i + 2]]
    target = text[i]
    data.append((context, target))

class CBOW(torch.nn.Module):
    def __init__(self, word_size, embedding_dim):
        super(CBOW, self).__init__()

        self.embeddings = nn.Embedding(word_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.activation_function1 = nn.ReLU()

        self.linear2 = nn.Linear(128, word_size)
        self.activation_function2 = nn.LogSoftmax(dim = -1)

    def forward(self, inputs):
        embeds = sum(self.embeddings(inputs)).view(1,-1)
        out = self.linear1(embeds)
        out = self.activation_function1(out)
        out = self.linear2(out)
        out = self.activation_function2(out)
        return out

    def get_word_emdedding(self, word):
        word = torch.tensor([word_to_ix[word]])
        return self.embeddings(word).view(1,-1)

model = CBOW(word_size, EMDEDDING_DIM)

loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

#开始训练
for epoch in range(100):
    total_loss = 0

    for context, target in data:
        context_vector = make_context_vector(context, word_to_ix)

        log_probs = model(context_vector)

        total_loss += loss_function(log_probs, torch.tensor([word_to_ix[target]]))
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

#预测
context1 = ['preparation','for','is', 'doing']
context_vector1 = make_context_vector(context1, word_to_ix)
a = model(context_vector1)

context2 = ['People','who', 'loved', 'once']
context_vector2 = make_context_vector(context2, word_to_ix)
b = model(context_vector2)

print(f'文本数据: {" ".join(text)}\n')
print(f'预测1: {context1}\n')
print(f'预测结果: {ix_to_word[torch.argmax(a[0]).item()]}')
print('\n')
print(f'预测2: {context2}\n')
print(f'预测结果: {ix_to_word[torch.argmax(b[0]).item()]}')
