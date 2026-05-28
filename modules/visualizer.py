import numpy as np
import matplotlib.pyplot as plt

def plot_overfitting_analysis(train_scores, test_scores, model_names):
    x = np.arange(len(model_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, train_scores, width, label='Train Score (F1)', color='#4c72b0')
    rects2 = ax.bar(x + width/2, test_scores, width, label='Test Score (F1)', color='#dd8452')
    
    ax.set_ylabel('F1-Score (Macro)')
    ax.set_title('Model Generalization & Overfitting Analysis')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend(loc='lower right')
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
                    
    plt.tight_layout()
    plt.show()