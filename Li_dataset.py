from transformers import (RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, DataCollatorForLanguageModeling, Trainer,
    TrainingArguments)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
import sys
import os
import yaml
from rdkit import Chem
from sklearn.preprocessing import OneHotEncoder
from copy import deepcopy
from packaging import version

dataset =  pd.read_csv('data/UjjData.csv')

class Downstream_Dataset(Dataset):
    def __init__(self, dataset, tokenizer, max_token_len):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_token_len = max_token_len

    def __len__(self):
        self.len = len(self.dataset)
        return self.len

    def __getitem__(self, i):
        data_row = self.dataset.iloc[i]
        sd1 = data_row.iloc[0]
        
        anion_smile = data_row.iloc[1]
        unique_AmSM = np.unique(dataset['Anion Smiles'].values)
        one_hot_enocder =  OneHotEncoder(categories=[unique_AmSM], sparse=False)
        one_hot_enocder.fit([[smiles] for smiles in unique_AmSM])
        anion_smile_encoding = torch.tensor(one_hot_enocder.transform([[anion_smile]])[0].astype(int))
        
        log_li = data_row.iloc[2]
        comonomer_percentage = data_row.iloc[3]/100
        approxMW = data_row.iloc[4]
        approxTg = data_row.iloc[5]
        
        chain_architecture = 1
        if data_row.iloc[6] == 'S_2':
            chain_architecture = 0
    
        prop = data_row.iloc[7]

        encoding = self.tokenizer(
            str(sd1),
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        

        return dict(
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            anion_smile_encoding = anion_smile_encoding,
            log_li = torch.tensor(log_li),
            comonomer_percentage = torch.tensor(comonomer_percentage),
            approxMW = torch.tensor(approxMW),
            approxTg = torch.tensor(approxTg),
            chain_architecture = torch.tensor(chain_architecture),
            prop=prop
    )
    
class DataAugmentation:
    def __init__(self, aug_indicator):
        super(DataAugmentation, self).__init__()
        self.aug_indicator = aug_indicator

    """Rotate atoms to generate more SMILES"""
    def rotate_atoms(self, li, x):
        return (li[x % len(li):] + li[:x % len(li)])

    """Generate SMILES"""
    def generate_smiles(self, smiles):
        smiles_list = []
        try:
            mol = Chem.MolFromSmiles(smiles)
        except:
            mol = None
        if mol != None:
            n_atoms = mol.GetNumAtoms()
            n_atoms_list = [nat for nat in range(n_atoms)]
            if n_atoms != 0:
                for iatoms in range(n_atoms):
                    n_atoms_list_tmp = self.rotate_atoms(n_atoms_list, iatoms)  # rotate atoms' index
                    nmol = Chem.RenumberAtoms(mol, n_atoms_list_tmp)  # renumber atoms in mol
                    try:
                        smiles = Chem.MolToSmiles(nmol,
                                                  isomericSmiles=True,  # keep isomerism
                                                  kekuleSmiles=False,  # kekulize or not
                                                  rootedAtAtom=-1,  # default
                                                  canonical=False,  # canonicalize or not
                                                  allBondsExplicit=False,  #
                                                  allHsExplicit=False)  #
                    except:
                        smiles = 'None'
                    smiles_list.append(smiles)
            else:
                smiles = 'None'
                smiles_list.append(smiles)
        else:
            try:
                smiles = Chem.MolToSmiles(mol,
                                          isomericSmiles=True,  # keep isomerism
                                          kekuleSmiles=False,  # kekulize or not
                                          rootedAtAtom=-1,  # default
                                          canonical=False,  # canonicalize or not
                                          allBondsExplicit=False,  #
                                          allHsExplicit=False)  #
            except:
                smiles = 'None'
            smiles_list.append(smiles)
        smiles_array = pd.DataFrame(smiles_list).drop_duplicates().values
        # """
        if self.aug_indicator is not None:
            smiles_aug = smiles_array[1:, :]
            np.random.shuffle(smiles_aug)
            smiles_array = np.vstack((smiles_array[0, :], smiles_aug[:self.aug_indicator-1, :]))
        return smiles_array

    """SMILES Augmentation"""
    def smiles_augmentation(self, df):
        column_list = df.columns
        data_aug = np.zeros((1, df.shape[1]))
        for i in range(df.shape[0]):
            smiles = df.iloc[i, 0]
            prop = df.iloc[i, 1:]
            smiles_array = self.generate_smiles(smiles)
            if 'None' not in smiles_array:
                prop = np.tile(prop, (len(smiles_array), 1))
                data_new = np.hstack((smiles_array, prop))
                data_aug = np.vstack((data_aug, data_new))
        df_aug = pd.DataFrame(data_aug[1:, :], columns=column_list)
        return df_aug

    """Used for copolymers with two repeating units"""
    def smiles_augmentation_2(self, df):
        df_columns = df.columns
        column_list = df.columns.tolist()
        column_list_temp = deepcopy(column_list)
        column_list_temp[0] = column_list[1]
        column_list_temp[1] = column_list[0]
        df = df[column_list_temp]
        data_aug = np.zeros((1, df.shape[1]))
        for i in range(df.shape[0]):
            if df.loc[i, "Comonomer percentage"] == 100.0:
                data_new = df.values[i, :].reshape(1, -1)
                data_aug = np.vstack((data_aug, data_new))
            else:
                smiles = df.iloc[i, 0]
                prop = df.iloc[i, 1:]
                smiles_array = self.generate_smiles(smiles)
                if 'None' not in smiles_array:
                    prop = np.tile(prop, (len(smiles_array), 1))
                    data_new = np.hstack((smiles_array, prop))
                    data_aug = np.vstack((data_aug, data_new))
        data_aug_copy = deepcopy(data_aug)
        data_aug_copy[:, 0] = data_aug[:, 1]
        data_aug_copy[:, 1] = data_aug[:, 0]
        df_aug = pd.DataFrame(data_aug_copy[1:, :], columns=df_columns)
        return df_aug
    
    def combine_smiles(self, df):
        for i in range(df.shape[0]):
            if df.loc[i, "Comonomer percentage"] != 100.0:
                df.loc[i, "SMILES descriptor 1"] = df.loc[i, "SMILES descriptor 1"] + '.' + df.loc[
                    i, "SMILES descriptor 2"]
        df = df.drop(columns=['SMILES descriptor 2'])
        return df