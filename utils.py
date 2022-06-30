import torch.nn as nn
import torch
import warnings as wrn
wrn.filterwarnings('ignore')


def error_df(df, error='Cognitive Error'):
    df = df.loc[df['ErrorType'] == error]
    df['Word'] = df['Word'].apply(word2char)
    df['Error'] = df['Error'].apply(word2char)
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.iloc[:, [1, 0]]
    df.to_csv('./Dataset/error.csv', index=False)


def word2char(word):
    w2c = [char for char in word]
    return ' '.join(w2c)


def basic_tokenizer(text):
    return text.split()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def save_model(model, train_loss, epoch, PATH):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss
    }, PATH)
    print(f"---------\nModel Saved at {PATH}\n---------\n")


def load_model(model, PATH):
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_loss = checkpoint['loss']
    return checkpoint, epoch, train_loss


if __name__ == '__main__':
    pass
