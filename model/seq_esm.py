import torch
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from esm import ESM2
import tqdm

def main(input_file, out_file, batchsize=32):
    df = pd.read_pickle(input_file)
    df = get_protein_embedding(df, batchsize=batchsize)
    df.to_pickle(out_file)


def get_protein_embedding(test_df: pd.DataFrame, batchsize):
    """get embedding for each protein.

    Args:
        test_df (pd.DataFrame): protein sequence DataFrame

    Returns:
        pd.DataFrame: protein embedding DataFrame
    """

    # dataset = EsmDataset(test_df)
    loader = DataLoader(dataset,
                        batch_size=batchsize,
                        num_workers=4,
                        collate_fn=dataset.collate_fn)

    backbone = dataset.model
    for p in backbone.parameters():
        p.requires_grad = False
    backbone = backbone.cuda()

    batch_mean_list = []
    for batch in tqdm(loader):
        batch = {key: val.cuda() for key, val in batch.items()}
        batch_aa_embeddings = backbone(batch['input_ids'],
                                       repr_layers=[33])['representations'][33]
        batch_aa_embeddings = batch_aa_embeddings[:, 1:, :]
        masked_lengths = batch['masked_lengths']
        batch_aa_embeddings = batch_aa_embeddings * masked_lengths.unsqueeze(
            -1)
        batch_mean = batch_aa_embeddings.sum(
            dim=1, keepdim=False) / masked_lengths.sum(1).unsqueeze(-1)
        batch_mean = batch_mean.detach().cpu()
        batch_mean_list.append(batch_mean)

        del batch, batch_aa_embeddings, batch_mean
        torch.cuda.empty_cache()

    mean_all = torch.cat(batch_mean_list, dim=0)
    test_df['embeddings'] = list(mean_all.numpy())
    return test_df
