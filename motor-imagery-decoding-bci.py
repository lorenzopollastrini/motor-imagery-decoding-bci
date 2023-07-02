# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP

# #############################################################################

# Impostazione dei parametri
tmin, tmax = -1.0, 4.0
event_id = dict(hands = 2, feet = 3)
subject = 2
runs = [6, 10, 14]  # Immaginazione del movimento: mani vs piedi

# Lettura dei dati
raw_fnames = eegbci.load_data(subject, runs)
raw = concatenate_raws([read_raw_edf(f, preload = True) for f in raw_fnames])
eegbci.standardize(raw)  # Impostazione dei nomi dei canali
montage = make_standard_montage("standard_1005")
raw.set_montage(montage)

# Applicazione di un filtro passa banda
raw.filter(7.0, 30.0, fir_design = "firwin", skip_by_annotation = "edge")

# Estrazione degli eventi dalle annotazioni
events, _ = events_from_annotations(raw, event_id = dict(T1 = 2, T2 = 3))

picks = pick_types(raw.info, meg = False, eeg = True, stim = False, eog = False, exclude = "bads")

# Lettura delle epoche
# La fase di test sarà effettuata con delle finestre scorrevoli sulle epoche
epochs = Epochs(
    raw,
    events,
    event_id,
    tmin,
    tmax,
    proj = True,
    picks = picks,
    baseline = None,
    preload = True,
)

# L'addestramento sarà effettuato solo considerando l'intervallo tra 1s e 2s di
# ciascuna epoca
epochs_train = epochs.copy().crop(tmin = 1.0, tmax = 2.0)

# Estrazione delle label e mappatura delle label ai valori 0 e 1
labels = epochs.events[:, -1] - 2

epochs_data = epochs.get_data()
epochs_data_train = epochs_train.get_data()

# Definizione di un cross-validator monte-carlo (per effettuare la media dei
# punteggi dei vari split e non risentire della varianza del punteggio di
# ciascuno split)
scores = []
cv = ShuffleSplit(10, test_size = 0.2)
cv_split = cv.split(epochs_data_train)

# Assemblaggio di un classificatore pipeline
csp = CSP()
lda = LinearDiscriminantAnalysis()
clf = Pipeline([("CSP", csp), ("LDA", lda)])

# Ricerca dell'iperparametro "n_components" ottimo per la CSP (tramite una grid
# search di scikit-learn basata sul cross-validator definito)
param_grid = {
    'CSP__n_components': range(1, 65)
    }
search = GridSearchCV(clf, param_grid, cv = cv).fit(epochs_data_train, labels)

# Punteggio di classificazione sul training set
training_set_score = search.score(epochs_data_train, labels)

y_pred = search.predict(epochs_data)

# Metriche di performance (di default la classe 0, "hands", è la classe dei
# negativi, mentre la classe 1, "feet", è la classe dei positivi)
# Accuracy (punteggio di classificazione sul test set)
accuracy = accuracy_score(labels, y_pred)
# Matrice di confusione
conf_matrix = confusion_matrix(labels, y_pred)
precision = precision_score(labels, y_pred)
recall = recall_score(labels, y_pred)
f1 = f1_score(labels, y_pred)

# Impostazione del miglior n_components per la CSP nella pipeline
# clf.set_params(**search.best_params_)
# clf.fit(epochs_data_train, labels)

# Mappe topografiche dei pattern spaziali della CSP addestrata sui dati
# completi
# Impostazione del miglior n_components per la CSP fuori dalla pipeline
csp.set_params(**{'n_components': search.best_params_.get('CSP__n_components')})
csp.fit(epochs_data, labels)
csp.plot_patterns(epochs.info, ch_type = "eeg", units = "Patterns (AU)", size = 1.5)

# *** GRAFICO CHE MOSTRA L'ANDAMENTO DELLA ACCURACY RISPETTO AL CENTRO DELLA
# FINESTRA TEMPORALE CHE SI USA PER CLASSIFICARE UN'EPOCA ***

sfreq = raw.info["sfreq"]
w_length = int(sfreq * 0.5)  # Lunghezza della finestra scorrevole
w_step = int(sfreq * 0.1)  # Step size della finestra scorrevole
# Istanti di inizio della finestra scorrevole
w_starts = np.arange(0, epochs_data.shape[2] - w_length, w_step)

# Punteggi per ogni split dello shuffle split
split_scores = []

for train_idx, test_idx in cv_split:
    y_train, y_test = labels[train_idx], labels[test_idx]

    X_train = csp.fit_transform(epochs_data_train[train_idx], y_train)

    lda.fit(X_train, y_train)

    # Test del classificatore sulla finestra scorrevole
    score_this_window = []
    for n in w_starts:
        X_test = csp.transform(epochs_data[test_idx][:, :, n : (n + w_length)])
        score_this_window.append(lda.score(X_test, y_test))
    split_scores.append(score_this_window)

# Ascisse del grafico
w_times = (w_starts + w_length / 2.0) / sfreq + epochs.tmin

plt.figure()
plt.plot(w_times, np.mean(split_scores, 0), label="Accuracy")
plt.axvline(0, linestyle="--", color="k", label="Onset")
plt.xlabel("Tempo (s)")
plt.ylabel("Accuracy")
plt.title("Accuracy nel tempo")
plt.legend(loc="lower right")
plt.show()

# Stampa dei risultati
print(
      "\n",
      "Punteggio di classificazione sul training set: %f\n" % (training_set_score),
      "Accuracy: %f\n" % (accuracy),
      "Matrice di confusione:\n", conf_matrix, "\n",
      "Precision: %f\n" % (precision),
      "Recall: %f\n" % (recall),
      "F1 score: %f" % (f1)
)