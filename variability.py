import os
import numpy as np
import librosa
import scipy.stats as stats

# Dossier principal où se trouvent tous les dossiers de classes
root_dir = 'data/yemba/train'

def compute_variance_within_class(mfccs):
    # Concaténer toutes les matrices MFCCs en une seule grande matrice
    all_mfccs = np.concatenate(mfccs, axis=1)
    # Calculer la variance de chaque coefficient MFCC (en moyenne sur les trames)
    variance = np.var(all_mfccs, axis=1)
    return variance

# Fonction pour zero-padding des MFCCs pour qu'ils aient tous la même longueur
def pad_mfccs(mfccs, max_len=None):
    if max_len is None:
        max_len = max(mfcc.shape[1] for mfcc in mfccs)
    padded_mfccs = []
    for mfcc in mfccs:
        if mfcc.shape[1] < max_len:
            pad_width = max_len - mfcc.shape[1]
            padded_mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            padded_mfcc = mfcc
        padded_mfccs.append(padded_mfcc)
    return np.array(padded_mfccs)

# Calculer les MFCCs pour chaque classe comme avant
mfccs_grouped = {}
# Dictionnaire pour stocker les variances des MFCCs par classe
variances_grouped = {}
for class_name in os.listdir(root_dir):
    class_path = os.path.join(root_dir, class_name)
    if os.path.isdir(class_path):
        mfccs = []
        for file_name in os.listdir(class_path):
            file_path = os.path.join(class_path, file_name)
            if file_name.endswith('.wav'):
                y, sr = librosa.load(file_path, sr=None)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfccs.append(mfcc)
        # Appliquer le zero-padding
        mfccs_grouped[class_name] = pad_mfccs(mfccs)
        # Calculer la variance des MFCCs à l'intérieur de la classe
        variance = compute_variance_within_class(mfccs)
        variances_grouped[class_name] = variance

# Supposons que nous voulons comparer la première dimension des MFCCs entre les classes
mfcc1_classes = {class_name: mfccs[:, 0, :].flatten() for class_name, mfccs in mfccs_grouped.items()}

# Effectuer l'ANOVA sur la première dimension des MFCCs
f_value, p_value = stats.f_oneway(*mfcc1_classes.values())

# Afficher les résultats
print("F-value:", f_value)
print("P-value:", p_value)

# Interprétation des résultats
if p_value < 0.05:
    print("Les variances des MFCC diffèrent significativement entre les classes.")
else:
    print("Aucune différence significative des variances des MFCC entre les classes.")
    
    
    
# Afficher les variances pour chaque classe
for class_name, variance in variances_grouped.items():
    print(f'Classe: {class_name}')
    print(f'Variance des MFCCs: {variance}')
    print()

