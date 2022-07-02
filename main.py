from utils import (
    word2char, basic_tokenizer, count_parameters, initialize_weights,
    save_model, load_model, error_df, train_valid_test_df,
)
from transformer import (
    Encoder, EncoderLayer, MultiHeadAttentionLayer,
    PositionwiseFeedforwardLayer, Decoder, DecoderLayer,
    Seq2Seq
)
from pipeline import train, evaluate
from metrics import evaluation_report

import pandas as pd
from sklearn.model_selection import train_test_split
from torchtext.legacy.data import Field, TabularDataset, BucketIterator
import torch
import torch.nn as nn
import os
import gc
import warnings as wrn
wrn.filterwarnings('ignore')


def main():
    df = pd.read_csv('./Dataset/sec_dataset_III_v2.csv')
    df_copy = df.copy()
    df['Word'] = df['Word'].apply(word2char)
    df['Error'] = df['Error'].apply(word2char)
    # df = df.sample(frac=1).reset_index(drop=True)
    # df = df.iloc[:, [1, 0]]
    # # print(df)

    # df = df.iloc[:5000, :]

    train_df, valid_df, test_df = train_valid_test_df(df, test_size=.15, valid_size=.05)

    # train_df, test_df = train_test_split(df, test_size=.15)
    # train_df, valid_df = train_test_split(train_df, test_size=.05)
    # print(len(train_df), len(valid_df), len(test_df))
    train_df.to_csv('./Dataset/train.csv', index=False)
    valid_df.to_csv('./Dataset/valid.csv', index=False)
    test_df.to_csv('./Dataset/test.csv', index=False)

    SRC = Field(
        tokenize=basic_tokenizer, lower=False,
        init_token='<sos>', eos_token='<eos>', batch_first=True
    )
    TRG = Field(
        tokenize=basic_tokenizer, lower=False,
        init_token='<sos>', eos_token='<eos>', batch_first=True
    )
    fields = {
        'Error': ('src', SRC),
        'Word': ('trg', TRG)
    }

    train_data, valid_data, test_data = TabularDataset.splits(
        path='./Dataset',
        train='train.csv',
        validation='valid.csv',
        test='test.csv',
        format='csv',
        fields=fields
    )

    SRC.build_vocab(train_data, min_freq=100)
    TRG.build_vocab(train_data, min_freq=50)
    # print(len(SRC.vocab), len(TRG.vocab))

    # ------------------------------
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 512

    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    HID_DIM = 128
    ENC_LAYERS = 3
    DEC_LAYERS = 3
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 256
    DEC_PF_DIM = 256
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1
    CLIP = 1
    N_EPOCHS = 10
    LEARNING_RATE = 0.0005
    PATH = './Checkpoints/transformer_v2.pth'
    # ------------------------------
    gc.collect()
    torch.cuda.empty_cache()
    # ------------------------------

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=DEVICE
    )

    enc = Encoder(
        INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM,
        ENC_DROPOUT, DEVICE
    )
    dec = Decoder(
        OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM,
        DEC_DROPOUT, DEVICE
    )

    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, DEVICE).to(DEVICE)
    model.apply(initialize_weights)
    # print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    epoch = 1
    best_loss = 1e10
    if os.path.exists(PATH):
        checkpoint, epoch, train_loss = load_model(model, PATH)
        best_loss = train_loss
    N_EPOCHS = epoch + 0

    for epoch in range(epoch, N_EPOCHS):
        print(f"Epoch: {epoch} / {N_EPOCHS}")
        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        print(f"Train Loss: {train_loss:.4f}")
        if train_loss < best_loss:
            best_loss = train_loss
            save_model(model, train_loss, epoch, PATH)

    # ---------------------
    # error_types = [
    #     'Cognitive Error', 'Homonym Error', 'Run-on Error',
    #     'Split-word Error (Left)', 'Split-word Error (Random)',
    #     'Split-word Error (Right)', 'Split-word Error (both)',
    #     'Typo (Avro) Substituition', 'Typo (Bijoy) Substituition',
    #     'Typo Deletion', 'Typo Insertion', 'Typo Transposition',
    #     'Visual Error', 'Visual Error (Combined Character)'
    # ]
    error_types = [
        'Homonym Error',
        'Typo Deletion',
        'Typo (Avro) Substituition',
        'Typo (Bijoy) Substituition',
        'Cognitive Error',
        'Run-on Error',
        'Split-word Error (Left)',
        'Split-word Error (Random)',
        'Split-word Error (Right)',
        'Split-word Error (both)',
        'Typo Insertion',
        'Typo Transposition',
        'Visual Error',
        'Visual Error (Combined Character)'
    ]

    for error_name in error_types:
        print(f'------\nError Type: {error_name}\n------')
        error_df(df_copy, error_name)

        error_data, _ = TabularDataset.splits(
            path='./Dataset',
            train='error.csv',
            test='error.csv',
            format='csv',
            fields=fields
        )

        eval_df = evaluation_report(error_data, SRC, TRG, model, DEVICE)

        error_name = error_name.replace(' ', '').replace('(', '').replace(')', '')
        # eval_df.to_csv(f'./Dataframes/transformer_{error_name}_v2.csv')
        print('\n\n')
    # ---------------------


if __name__ == '__main__':
    main()
