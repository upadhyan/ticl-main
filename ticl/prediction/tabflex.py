
from ticl.prediction.tabpfn import TabPFNClassifier
import numpy as np
import pdb
import torch
from ticl.utils import fetch_model


class TabFlex:
    def __init__(
        self,
    ):
        torch.set_num_threads(1)
        fetch_model('ssm_tabpfn_b4_maxnumclasses100_modellinear_attention_numfeatures1000_n1024_validdatanew_warm_08_23_2024_19_25_40_epoch_3140.cpkt')
        fetch_model('ssm_tabpfn_b4_largedatasetTrue_modellinear_attention_nsamples50000_08_01_2024_22_05_50_epoch_110.cpkt')
        fetch_model('ssm_tabpfn_modellinear_attention_08_28_2024_19_00_44_epoch_3110.cpkt')


        self.tabflexh1k = TabPFNClassifier(
            device='cuda', 
            model_string = f'ssm_tabpfn_b4_maxnumclasses100_modellinear_attention_numfeatures1000_n1024_validdatanew_warm_08_23_2024_19_25_40',
            N_ensemble_configurations=3,
            epoch = '3140',
        )

        self.tabflexl100 = TabPFNClassifier(
            device='cuda', 
            model_string = f'ssm_tabpfn_b4_largedatasetTrue_modellinear_attention_nsamples50000_08_01_2024_22_05_50',
            N_ensemble_configurations=1,
            epoch = '110', 
        )

        self.tabflexs100 = TabPFNClassifier(
            device='cuda', 
            model_string = f'ssm_tabpfn_modellinear_attention_08_28_2024_19_00_44',
            N_ensemble_configurations=3,
            epoch = '3110',
        )

    def fit(self, X, y):
        N, D = X.shape

        if N >= 3000 and D <= 100:
            self.model = self.tabflexl100
        elif D > 100 or (D/N >= 0.2 and N >= 3000):
            if D <= 1000:
                self.model = self.tabflexh1k
            else:
                self.model = self.tabflexh1k
                self.model.dimension_reduction = 'random_proj'
                self.model.fit(X, y, overwrite_warning=True)
                return self
        else:
            self.model = self.tabflexs100

        self.model.fit(X, y, overwrite_warning=True)

        return self

    def predict(self, X):
        y = self.model.predict(X)
        return y
