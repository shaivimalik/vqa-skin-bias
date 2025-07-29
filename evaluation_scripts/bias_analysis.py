import os
import pandas as pd
from tqdm import tqdm
from scipy.stats import f_oneway, ttest_ind

words = pd.read_csv(os.path.join('..','selected_trait_words.csv'))['WORD_ENGLISH'].to_list()
output_dir = os.path.join('..', 'evaluation_results')
os.makedirs(output_dir, exist_ok=True)
datasets = ['fairface', 'aiface']
models = ['paligemma2-3b-mix-224', 'Phi-4-multimodal-instruct', 'llava-1.5-7b-hf', 'Qwen2.5-VL-3B-Instruct', 'blip2-opt-2.7b']

# bias analysis
for model in tqdm(models):

    for dataset in datasets:
        meta_df = pd.read_csv(os.path.join('..',f'selected_images_{dataset}.csv'))
        res_dir = os.path.join('..', f'{model}_results')
        if dataset=='fairface':
            group_columns = ['race', 'age', 'gender']
        else: group_columns =['Skin Tone']
        all_results = {attr: [] for attr in group_columns}

        for file in os.listdir(res_dir):
            if file.split(".csv")[0].split("_")[-1] == dataset:
                df = pd.read_csv(os.path.join(res_dir, file))
                df['file'] = df['img']
                merged_df = pd.merge(meta_df, df, on='file')

                for attr in group_columns:
                    grouped = merged_df.groupby(attr)
                    result = {
                            'trait': os.path.basename(file).split("_")[-2],
                            'attribute': attr
                    }

                    for template in range(5):
                        prob_grouped = grouped[f'yes_prob_{template}'].apply(list)
                        if attr == 'gender':
                            g1, g2 = prob_grouped.iloc[0], prob_grouped.iloc[1]
                            stat, pval = ttest_ind(g1, g2, equal_var=False)
                            test_type = 'ttest'
                        else:
                            stat, pval = f_oneway(*prob_grouped, equal_var=False) #f_oneway
                            test_type = 'anova'

                        overall_mean = merged_df[f'yes_prob_{template}'].mean()
                        result[f'overall_mean_template_{template}'] = overall_mean
                        result[f'p_value_template_{template}'] = pval
                        result[f'statistic_template_{template}'] = stat
                        result[f'test_type_template_{template}'] = test_type

                        means = {}
                        diffs = {}

                        for group_name, group_df in grouped:
                            values = group_df[f'yes_prob_{template}'].tolist()
                            group_mean = sum(values) / len(values)
                            means[f"{attr}_{group_name}_mean_template_{template}"] = group_mean
                            diffs[f"{attr}_{group_name}_diff_from_overall_template_{template}"] = group_mean - overall_mean
                        
                        result.update(means)
                        result.update(diffs)

                    all_results[attr].append(result)

        for attr, results in all_results.items():
            df = pd.DataFrame(results)
            output_file = os.path.join(output_dir, f"{model}_{attr}_bias_analysis.csv")
            df.to_csv(output_file, index=False)
            print(f"Saved: {output_file}")
