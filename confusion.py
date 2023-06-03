from sklearn.metrics import confusion_matrix
import pandas as pd
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import seaborn as sns

class ConfusionMatrix:
    def __init__(self, true, pred):
        self.matrix = confusion_matrix(true.flatten(), pred.flatten())
        self.tn, self.fp, self.fn, self.tp = self.matrix.ravel()

        self.accuracy = (self.tp + self.tn) / (self.tn + self.fp + self.fn + self.tp)
        self.recall = self.tp / (self.tp + self.fn)
        self.precision = self.tp / (self.tp + self.fp)
        self.specificity = self.tn / (self.tn + self.fp)

        self.df_matrix = pd.DataFrame(self.matrix,
                                      index=['Real Negative', 'Real Positive'],
                                      columns=['Predicted Negative', 'Predicted Positive'])

    def __str__(self):
        return f"{'Accuracy:'.ljust(13, ' ')}{self.accuracy:.3f}\n" \
               f"{'Recall:'.ljust(13, ' ')}{self.recall:.3f}\n" \
               f"{'Precision'.ljust(13, ' ')}{self.precision:.3f}\n" \
               f"{'Specificity:'.ljust(13, ' ')}{self.specificity:.3f}\n"

    def heatmap(self, annot=True, fmt=".0f", norm=LogNorm(), cmap='crest', cbar_kws=None):
        if cbar_kws is None:
            cbar_kws = {'ticks': []}
        sns.heatmap(self.df_matrix, annot=annot, fmt=fmt, norm=norm, cmap=cmap, cbar_kws=cbar_kws)
        plt.show()