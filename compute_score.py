from sklearn.metrics import f1_score
import pandas as pd


def compute_metric(pred, gt):
  score = f1_score(gt, pred, average='macro')
  return score

GT_PATH = "./train_lables.csv"
# GT_PATH = "./data/private_info/private.csv" # либо "./data/private_info/public.csv"
SUBM_PATH = "./data/submission.csv"

if __name__ == "__main__":
  subm_df = pd.read_csv(SUBM_PATH, sep="\t")
  gt_df = pd.read_csv(GT_PATH, sep="\t")

  result_df = gt_df.merge(subm_df, how="inner", on=["image_name"])
  pred = result_df["label_id_y"].tolist() + [-1 for _ in range(len(gt_df) - len(result_df))]

  metric = compute_metric(pred, gt_df["label_id"].tolist())
  print(f"F1 macro: {metric}")
  
