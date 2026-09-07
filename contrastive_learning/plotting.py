"""Plotting helpers for the contrastive model.

Extracted from the contrastive notebooks: breed distribution, Equal Earth /
unit-sphere embedding projection, and KNN@3 confusion matrix.
"""

import numpy as np


def plot_distribution(y, class_names, title="Distribuzione dataset"):
    import matplotlib.pyplot as plt
    import seaborn as sns

    u, c  = np.unique(y, return_counts=True)
    idx   = np.argsort(c)[::-1]
    names = [class_names[i] for i in u[idx]]
    fw    = max(16, len(u)*0.45)
    plt.figure(figsize=(fw, 6))
    ax = sns.barplot(x=names, y=c[idx], palette='viridis')
    for i, v in enumerate(c[idx]):
        ax.text(i, v+0.3, str(v), ha='center', va='bottom', fontsize=7, rotation=90)
    plt.xticks(rotation=90, fontsize=8)
    plt.title(title, fontsize=13); plt.xlabel('Breed'); plt.ylabel('Samples')
    plt.tight_layout(); plt.show()


def plot_embedding(best_Z, best_y, class_names,
                   save_path="results/embedding_paperfaithful.png"):
    """Equal Earth projection (left) + unit 3-sphere (right) scatter."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    from .evaluation import equal_earth_projection

    Z2d = equal_earth_projection(best_Z)
    nc  = len(class_names)
    pal = sns.color_palette('tab20', nc)

    fig = plt.figure(figsize=(22, 8))

    # Equal Earth 2D
    ax1 = fig.add_subplot(1, 2, 1)
    for c in range(nc):
        m = best_y == c
        if m.any():
            ax1.scatter(Z2d[m,0], Z2d[m,1], c=[pal[c % 20]],
                        label=class_names[c], s=30, alpha=0.85, edgecolors='none')
    ax1.set_title('Equal Earth Projection (best fold)', fontsize=13)
    ax1.set_xlabel('x'); ax1.set_ylabel('y')
    if nc <= 30:
        ax1.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc='upper left')

    # Sfera 3D
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    for c in range(nc):
        m = best_y == c
        if m.any():
            ax2.scatter(best_Z[m,0], best_Z[m,1], best_Z[m,2],
                        c=[pal[c % 20]], s=20, alpha=0.7)
    ax2.set_title('Sfera unitaria 3D (best fold)', fontsize=13)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"💾 {save_path}")


def plot_confusion_matrix(best_Z, best_y, class_names,
                          save_path="results/cm_paperfaithful.png"):
    """KNN@3 confusion matrix (row-normalised percentages)."""
    import matplotlib.pyplot as plt

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import confusion_matrix

    knn  = KNeighborsClassifier(n_neighbors=3).fit(best_Z, best_y)
    yp   = knn.predict(best_Z)
    cm   = confusion_matrix(best_y, yp)

    MAX = 40
    nc = len(class_names)
    if nc > MAX:
        imp = cm.sum(0) + cm.sum(1)
        top = np.argsort(imp)[-MAX:][::-1]
        cm, names = cm[np.ix_(top,top)], np.asarray(class_names)[top]
    else:
        names = class_names

    cm_n = cm.astype(float) / (cm.sum(1, keepdims=True) + 1e-10) * 100
    fs   = max(16, len(names)*0.55)
    tf   = max(7,  14 - len(names)//8)

    fig, ax = plt.subplots(figsize=(fs, fs))
    im = ax.imshow(cm_n, cmap='Blues', aspect='auto', vmin=0, vmax=100)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='%')
    ax.set(xticks=range(len(names)), yticks=range(len(names)),
           xticklabels=names, yticklabels=names)
    ax.set_title('Confusion Matrix KNN@3 — % normalizzata', fontsize=14)
    ax.set_ylabel('True'); ax.set_xlabel('Predicted')
    plt.setp(ax.get_xticklabels(), rotation=90, ha='right', fontsize=tf)
    plt.setp(ax.get_yticklabels(), fontsize=tf)
    for i in range(len(names)):
        for j in range(len(names)):
            v = cm_n[i,j]
            if v > 0:
                ax.text(j, i, f'{v:.0f}', ha='center', va='center',
                        fontsize=max(5, tf-2), color='white' if v > 50 else 'black')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"💾 {save_path}")
