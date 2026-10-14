"""Plot weighted class-error contributions from the saved September audit."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/sasol-oct14-mpl')
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
ap=argparse.ArgumentParser();ap.add_argument('--evidence',type=Path,default=Path(__file__).resolve().parents[2]/'outputs/next_round');ap.add_argument('--output',type=Path);a=ap.parse_args()
p=pd.read_csv(a.evidence/'class_error_contributions.csv');p=p[p.origin.eq('2025-09-01')].sort_values('weighted_f1_deficit',ascending=False).head(6)
fig,ax=plt.subplots(figsize=(8,4));labels=p.label.str.replace('Existing-category growth: ','Growth: ',regex=False);ax.barh(labels,p.weighted_f1_deficit,color='#176b87');ax.invert_yaxis();ax.set_xlabel('Contribution to 1 − weighted F1: support fraction × (1 − class F1)');ax.set_title('September baseline: fuel growth contributes the largest deficit');ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(a.output or a.evidence/'weighted_f1_deficit.png',dpi=180)
print('Saved weighted_f1_deficit audit plot')
