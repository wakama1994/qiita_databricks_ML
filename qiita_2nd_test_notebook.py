# Databricks notebook source
import pycaret

# COMMAND ----------

from pycaret.regression import * #回帰のインポート
from pycaret.datasets import get_data #データセットの読み込み
dataset = get_data("diamond") #ダイヤモンドのサンプルデータを取得

# COMMAND ----------

data = dataset.sample(frac=0.9, random_state=789) #学習データ frac:抽出する行の割合、random_state:乱数シード
data_unseen = dataset.drop(data.index) 
data.reset_index(drop=True, inplace=True)
data_unseen.reset_index(drop=True, inplace=True) #検証データ

# COMMAND ----------

exp1 = setup(data=data, target='Price', session_id=123, ignore_features=None) #前処理を実行

# COMMAND ----------

 best = compare_models() #モデル比較

# COMMAND ----------

model = create_model("et") #予測モデルに Extra Trees Regressor を選択

# COMMAND ----------

tuned_model = tune_model(model) #パラメータチューニング

# COMMAND ----------

evaluate_model(tuned_model) #パラメータ確認

# COMMAND ----------

plot_model(tuned_model, plot='error') #予測誤差プロット

# COMMAND ----------

plot_model(tuned_model, plot='feature') #特徴量プロット


# COMMAND ----------

final_model = finalize_model(tuned_model) #モデルの確定

# COMMAND ----------

result = predict_model(final_model, data = data_unseen) #テストデータで予測
result

# COMMAND ----------

import pandas as pd 
# databricks内のcsvデータ読み込み
dataset = pd.read_csv('/Volumes/{catalog_name}/{schema_name}/{table_name}/train.csv', encoding='utf-8')
data = dataset.sample(frac=0.9, random_state=789) #学習データ frac:抽出する行の割合、random_state:乱数シード
data_unseen = dataset.drop(data.index) 
data.reset_index(drop=True, inplace=True)
data_unseen.reset_index(drop=True, inplace=True) #検証データ
exp1 = setup(data=data, target='Survived', session_id=123, ignore_features=None) #前処理を実行
best = compare_models() #モデル比較

# COMMAND ----------

# databricksのテーブルからデータを読み込み
df= spark.read.table('{catalog_name}.{schema_name}.{table_name}')
dataset = df.select("*").toPandas()
data = dataset.sample(frac=0.9, random_state=789) #学習データ frac:抽出する行の割合、random_state:乱数シード
data_unseen = dataset.drop(data.index) 
data.reset_index(drop=True, inplace=True)
data_unseen.reset_index(drop=True, inplace=True) #検証データ
exp1 = setup(data=data, target='Survived', session_id=123, ignore_features=None) #前処理を実行
best = compare_models() #モデル比較
