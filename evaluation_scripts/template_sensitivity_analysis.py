import os
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare
from statsmodels.stats.anova import AnovaRM

def plot_template_sensitivity(output_path, words, res_dir, model, dataset):

    all_probs = []
    for word in words:
        filepath = os.path.join(res_dir, f"{model}_{word}_{dataset}.csv")
        sample = pd.read_csv(filepath)
        mean_probs = sample[[f'yes_prob_{i}' for i in range(5)]].mean().values
        all_probs.append(mean_probs)
    probabilities = np.array(all_probs).flatten() / 100

    plot_data = pd.DataFrame({
        'Trait Words': words * 5,
        'Templates': ['T1']*100 + ['T2']*100 + ['T3']*100 + ['T4']*100 + ['T5']*100,
        'Prob': probabilities
    })

    plt.figure(figsize=(21, 6))
    sns.set(style="whitegrid")
    sns.scatterplot(data=plot_data, x='Trait Words', y='Prob', hue='Templates', s=60)
    plt.xticks(rotation=90)
    plt.ylabel('Mean of P(Yes | image, trait, template)')
    plt.xlabel('Personality Trait')
    plt.ylim(0.2, 0.8)
    plt.legend(title='', loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5)
    plt.tight_layout()
    plt.gca().margins(x=0.015)
    plt.savefig(output_path, format='pdf', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"{model} template sensitivity plotted.")

def run_anova_and_friedman(res_dir):
    summary = []
    for file in os.listdir(res_dir):
        df = pd.read_csv(os.path.join(res_dir, file))
        if file.split(".csv")[0].split("_")[-1] == dataset:
          df_long = pd.melt(df,
                          id_vars=['img'],
                          value_vars=[col for col in df.columns if col.startswith('yes_prob_')],
                          var_name='template',
                          value_name='yes_prob')
          df_long['template'] = df_long['template'].str.replace('yes_prob_', '').astype(int)
          aov = AnovaRM(data=df_long, depvar='yes_prob', subject='img', within=['template'])
          res_anova = aov.fit()
          anova_row = res_anova.anova_table.loc['template']
          df_wide = df_long.pivot(index='img', columns='template', values='yes_prob')
          df_wide = df_wide.sort_index(axis=1)
          stat, pval = friedmanchisquare(*[df_wide[col] for col in df_wide.columns])
          summary.append({
          'file': file.split("_")[1],
          'ANOVA_F': anova_row['F Value'],
          'ANOVA_NumDF': anova_row['Num DF'],
          'ANOVA_DenDF': anova_row['Den DF'],
          'ANOVA_p': anova_row['Pr > F'],
          'Friedman_stat': stat,
          'Friedman_p': pval
          })
          print(f"Processed {file} successfully.")
    return pd.DataFrame(summary)

words = pd.read_csv(os.path.join('..','selected_trait_words.csv'))['WORD_ENGLISH'].to_list()
output_dir = os.path.join('..', 'evaluation_results')
os.makedirs(output_dir, exist_ok=True)
output_dir_plot = os.path.join('..', 'evaluation_plots')
os.makedirs(output_dir_plot, exist_ok=True)
dataset = 'fairface'
models = ['Phi-4-multimodal-instruct', 'blip2-opt-2.7b', 'Qwen2.5-VL-3B-Instruct', 'llava-1.5-7b-hf','paligemma2-3b-mix-224']

for model in tqdm(models, desc="Processing model"):
    res_dir = os.path.join('.', f'{model}_results')
    plot_path = os.path.join(output_dir, f"template_sensitivity_{model}.pdf")
    plot_template_sensitivity(plot_path, words, res_dir, model, dataset)
    df = run_anova_and_friedman(res_dir)
    df.to_csv(os.path.join(output_dir, f'template_sensitivity_results_{model}.csv'), index=False)

print("Template sensitivity analysis complete.")