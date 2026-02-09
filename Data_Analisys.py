import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import MSTL
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

#Importando os dados
df = pd.read_csv("MiningProcess_Flotation_Plant_Database.csv")
df.info()
df.head()

# Transforma a data em datetime

df['date'] = pd.to_datetime(df['date'])

# Transforma as variáveis em float

for col in df.columns[1:]:
    df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
df.info()

# Plotando a variável alvo

TARGET = '% Silica Concentrate'

plt.figure(figsize=(15,5))
plt.plot(df['date'], df[TARGET])
plt.title('Silica Concentrate (%)')
plt.xlabel('Data')
plt.ylabel('% Silica')
plt.savefig("graphs/silica_plot.png", dpi=300, bbox_inches="tight")
plt.show()

# Checando gaps nas horas


# conta quantas vezes cada data (ano/mes/dia/hora) se repete
hourly_counts = (
    df.set_index('date')
      .resample('h')
      .size()
      .rename('count')
)

# os dados são medidos a cada 20 segundos, portanto:
EXPECTED_PER_HOUR = 60*60/20

# obtendo as horas diferentes do esperado
hourly_anomalies = hourly_counts[hourly_counts != EXPECTED_PER_HOUR]

# informações relevantes
total_hours = len(hourly_counts)
num_anomalous_hours = len(hourly_anomalies)
num_missing_hours = (hourly_counts == 0).sum()

# printando as informações 
print(f"Total de horas: {total_hours}")
print(f"# Horas anômalas: {num_anomalous_hours} (das quais {num_missing_hours} estão faltando)")
print("Horas anômalas:")
print(hourly_anomalies)

# Excluindo dados anteriores ao gap 

cutoff = pd.Timestamp('2017-03-29 12:00:00')
df = df[df['date'] >= cutoff]

# Adicionando uma linha no Timestamp com 179 (2017-04-10 00:00:00)

target_time = pd.Timestamp('2017-04-10 00:00:00')

# obtendo a primeira posição com o target_time
mask = (df['date'] == target_time).to_numpy()
pos = np.flatnonzero(mask)
p = pos[0]

# duplicando a linha
df = pd.concat([df.iloc[:p+1], df.iloc[[p]], df.iloc[p+1:]], axis=0)

# conta quantas vezes cada data (ano/mes/dia/hora) se repete
hourly_counts = (
    df.set_index('date')
      .resample('h')
      .size()
      .rename('count')
)

# obtendo as horas diferentes do esperado
hourly_anomalies = hourly_counts[hourly_counts != EXPECTED_PER_HOUR]

# informações relevantes
total_hours = len(hourly_counts)
num_anomalous_hours = len(hourly_anomalies)


# printando as informações 
print(f"Total de horas: {total_hours}")
print(f"# Horas anômalas: {num_anomalous_hours}")

# Distinguindo variáveis estáticas e dinâmicas  

cols = [c for c in df.columns if c != 'date']


# # de valores únicos de cada coluna em cada data
per_hour_nunique = (
    df
    .groupby('date')[cols]
    .nunique()
)

# separando variáveis estáticas e dinâmicas 
static_every_hour = per_hour_nunique.max(axis=0) == 1
not_static_every_hour  = per_hour_nunique.max(axis=0) > 1

constant_cols = static_every_hour[static_every_hour].index.tolist()
varying_cols = not_static_every_hour[not_static_every_hour].index.tolist()

# # de horas que cada coluna varia
varies_in_how_many_hours = (per_hour_nunique > 1).sum(axis=0).sort_values(ascending=False)


# plotando
summary_df = pd.DataFrame({
    "column": cols,
    "n_hours_varying": varies_in_how_many_hours.reindex(cols).values
})

plt.figure(figsize=(14, 6), dpi=100)
sns.barplot(
    data=summary_df.sort_values("n_hours_varying", ascending=False),
    x="column",
    y="n_hours_varying",
    dodge=False,
)

plt.title("Quantas horas cada parâmetro varia")
plt.xlabel("Parâmetro")
plt.ylabel("Horas com Variação")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("graphs/hours_varying.png", dpi=300, bbox_inches="tight")
plt.show()

# Investigando as horas em que '% Iron concentrate' e '% Silica concentrate' não são estáticas

IRON = '% Iron Concentrate'

# obtem as horas que as colunas variam e sua intersessão
iron_varying_hours = per_hour_nunique.index[per_hour_nunique[IRON] > 1]
silica_varying_hours = per_hour_nunique.index[per_hour_nunique[TARGET] > 1]
both_vary_hours = iron_varying_hours.intersection(silica_varying_hours)

#printando e plotando
print(f"{IRON} varia em {len(iron_varying_hours)} horas")
print(f"{TARGET} varia em {len(silica_varying_hours)} horas")
print(f"Ambos variam em {len(both_vary_hours)} horas")

fig, axes = plt.subplots(3, 1, figsize=(14, 9), dpi=100, sharex=True)

# Ferro 
axes[0].scatter(iron_varying_hours, [0]*len(iron_varying_hours), s=20, color='blue')
axes[0].set_title(f"Horas em que '{IRON}' varia")
axes[0].set_yticks([])

# Sílica 
axes[1].scatter(silica_varying_hours, [0]*len(silica_varying_hours), s=20, color='red')
axes[1].set_title(f"Horas em que '{TARGET}' varia")
axes[1].set_yticks([])

# Ambos
axes[2].scatter(both_vary_hours, [0]*len(both_vary_hours), s=20, color='purple')
axes[2].set_title("Horas em que ambos variam")
axes[2].set_yticks([])

plt.xlabel("Data")
plt.tight_layout()
plt.savefig("graphs/hours_varying_iron_silica.png", dpi=300, bbox_inches="tight")
plt.show()

# Exemplos de horas em que '% Iron concentrate' varia

example_iron_hours = iron_varying_hours[:: max(1, len(iron_varying_hours)//3)]

for i, h in enumerate(example_iron_hours):

    # hora em que a coluna varia
    subset = df[df['date'] == h]
    
    # plota a variação
    plt.figure(figsize=(12,4), dpi=100)
    plt.plot(subset.index, subset[IRON], marker='o', lw=1)
    plt.title(f"{ IRON} —  {h}")
    plt.xlabel("Timestamp (within hour)")
    plt.ylabel(IRON)
    plt.tight_layout()
    plt.savefig(f"graphs/example_hours_iron_varying_{i+1}.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Exemplos de horas em que '% Silica concentrate' varia

example_silica_hours = silica_varying_hours[:: max(1, len(silica_varying_hours)//3)]


for i, h in enumerate(example_silica_hours):

    # hora em que a coluna varia
    subset = df[df['date'] == h]
    
    # plota a variação
    plt.figure(figsize=(12,4), dpi=100)
    plt.plot(subset.index, subset[ TARGET], marker='o', lw=1, color = 'red')
    plt.title(f"{ TARGET} —  {h}")
    plt.xlabel("Timestamp (within hour)")
    plt.ylabel( TARGET)
    plt.tight_layout()
    plt.savefig(f"graphs/example_hours_silica_varying_{i+1}.png", dpi=300, bbox_inches="tight")
    plt.show()

# Define as variáveis '% Silica concentrate' e '% Iron concentrate' como estàticas em vez de dinâmicas

varying_cols.remove(TARGET)
varying_cols.remove(IRON)
constant_cols.append(TARGET)
constant_cols.append(IRON)

# printando variáveis estáticas e dinâmicas
print("\nParameters constant within EVERY complete hour (hourly/static fields):")
print(constant_cols)
print("\nParameters varying within complete hours (high-frequency sensors):")
print(varying_cols)

# Tranforma o Dataframe em horário 

target_cols = [IRON, TARGET]

# a data se torna o indice
df = df.set_index('date')

# intervalos de uma hora fechados à esquerda
rule = 'h'
kw   = dict(label='left', closed='left')


# média das variáveis dinâmicas, primeiro valor das variáveis concentrate e com delay para feed
varying_mean   = df[varying_cols].resample(rule, **kw).mean().shift(1)
feed_variables = df[constant_cols].resample(rule, **kw).first().shift(1).drop(columns=target_cols)
concentrate_variables = df[constant_cols].resample(rule, **kw).first()[target_cols]


# concatena as três partes
df_hourly = pd.concat([feed_variables, varying_mean, concentrate_variables], axis=1)

df_hourly.head()

df_hourly['% Silica Concentrate Delta'] = df_hourly[TARGET].diff().shift(1)

df_hourly.dropna(inplace=True)

df_hourly.head()

print(df_hourly['% Iron Feed'].corr(df_hourly['% Silica Feed']))
print(df_hourly['% Iron Concentrate'].corr(df_hourly['% Silica Concentrate']))

correlations = df_hourly[['% Iron Feed', '% Silica Feed', '% Silica Concentrate']].corr()

print(correlations['% Silica Concentrate'])

df_hourly.drop(columns=['% Silica Feed', '% Iron Concentrate'], inplace=True)

# Plotando a variável alvo

plt.figure(figsize=(15,5))
plt.plot(df_hourly.index, df_hourly[TARGET], alpha=0.4, label='Raw Hourly Mean')
df_hourly['silica_smooth'] = df_hourly[TARGET].rolling(window=24, min_periods=1).mean()
plt.plot(df_hourly.index, df_hourly['silica_smooth'], color='red', label='24-Hour Rolling Mean')
plt.title('Final Silica Concentrate (%) Over Time (Hourly Aggregation)')
plt.xlabel('Date')
plt.ylabel('% Silica')
plt.legend()
plt.savefig("graphs/hourly_silica_plot.png", dpi=300, bbox_inches="tight")
plt.show()

df_hourly = df_hourly.drop(columns=['silica_smooth'])

# Features

AMINE = 'Amina Flow'
PULP_FLOW = 'Ore Pulp Flow'
PULP_DENSITY = 'Ore Pulp Density'
STARCH = 'Starch Flow'


df_hourly['True_Amine_Dosage'] = df_hourly[AMINE] / (df_hourly[PULP_FLOW] * df_hourly[PULP_DENSITY]).replace(0, np.nan)
df_hourly['True_Starch_Dosage'] = df_hourly[STARCH] / (df_hourly[PULP_FLOW] * df_hourly[PULP_DENSITY]).replace(0, np.nan)

# Calcula correlação com Silica

comparison_cols = [
    STARCH,
    AMINE,  
    'True_Amine_Dosage',
    'True_Starch_Dosage'
]

correlations = df_hourly[comparison_cols + [TARGET]].corr()[TARGET].drop(TARGET)
correlations = correlations.sort_values(ascending=False)

# Plotando as correlações

plt.figure(figsize=(12, 6))
colors = ['firebrick' if x < 0 else 'steelblue' for x in correlations.values]
correlations.plot(kind='barh', color=colors)
plt.axvline(0, color='black', linewidth=0.8)
plt.title('Impact of Physics-Informed Feature Engineering')
plt.xlabel('Correlation with % Silica Concentrate')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig("graphs/physics_feature_importance.png", dpi=300)
plt.show()

# Correlação das features novas com as originais

comp_starch = ['True_Starch_Dosage', STARCH]
comp_amina = ['True_Amine_Dosage', AMINE ]

check_cols = comp_starch + comp_amina
corr_matrix = df_hourly[check_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Teste de redundância")
plt.savefig("graphs/Redundancy Check.png", dpi=300)
plt.show()

# Dropando as variáveis originais

df_hourly.drop(columns=[AMINE], inplace=True)
df_hourly.drop(columns=[STARCH], inplace=True)

# Analisando a correlação entre as colunas de ar

air_cols = [c for c in cols if 'Air Flow' in c]

plt.figure(figsize=(8, 6))
sns.heatmap(df_hourly[air_cols].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlação Dentre as Colunas de Ar")
plt.savefig("graphs/columns_air_correlation.png", dpi=300, bbox_inches="tight")
plt.show()

# Agrupando colunas de ar altamente correlacionadas

group_1_cols = [air_cols[0], air_cols[1], air_cols[2]] # Cols 1, 2, 3 
group_2_cols = [air_cols[5], air_cols[6]]              # Cols 6, 7
col_4_name   = air_cols[3]                             # col 4
col_5_name   = air_cols[4]                             #col 5

df_hourly['Air_Cluster_123'] = df_hourly[group_1_cols].mean(axis=1)
df_hourly['Air_Cluster_67'] = df_hourly[group_2_cols].mean(axis=1)

# Realocando as colunas

col = df_hourly.pop('Air_Cluster_123')
df_hourly.insert(6, 'Air_Cluster_123', col)

col = df_hourly.pop('Air_Cluster_67')
df_hourly.insert(12, 'Air_Cluster_67', col)

# Dropando as colunas originais

cols_to_drop = group_1_cols + group_2_cols
df_hourly.drop(columns=cols_to_drop, inplace=True)

# Analisando a correlação entre o level das colunas

level_cols = [c for c in cols if 'Level' in c]

plt.figure(figsize=(8, 6))
sns.heatmap(df_hourly[level_cols].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlação entre o nível das Colunas")
plt.savefig("graphs/column_level_correlation.png", dpi=300, bbox_inches="tight")
plt.show()

# Agrupando level de colunas altamente correlacionados

level_cols_123 = [level_cols[0], level_cols[1], level_cols[2]]                 # cols 1, 2, 3
level_cols_4567 = [level_cols[3], level_cols[4], level_cols[5], level_cols[6]] # cols 4,5,6,7

df_hourly['Level_Cluster_123'] = df_hourly[level_cols_123].mean(axis=1)
df_hourly['Level_Cluster_4567'] = df_hourly[level_cols_4567].mean(axis=1)

# Realocando as colunas

col = df_hourly.pop('Level_Cluster_123')
df_hourly.insert(10, 'Level_Cluster_123', col)

col = df_hourly.pop('Level_Cluster_4567')
df_hourly.insert(11, 'Level_Cluster_4567', col)

# Dropando as colunas originais

df_hourly.drop(columns=level_cols_123 + level_cols_4567, inplace=True)

# Histogramas

cols = df_hourly.columns
n_cols = 4
n_rows = int(np.ceil(len(cols) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
axes = axes.flatten()
for i, col in enumerate(cols):
    ax = axes[i]
    data = df_hourly[col]
    ax.hist(data, bins=20, color='steelblue', edgecolor='black', linewidth=0.6)
    ax.set_title(col, fontsize=9)
    ax.set_xlabel('')
    ax.set_ylabel('Frequency')
    ax.axvline(data.mean(), color='red', linestyle='dashed', linewidth=1)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
fig.suptitle('Distributions of Variables', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("graphs/Histograms.png", dpi=300, bbox_inches="tight")
plt.show()

# Outliers

outlier_summary = []

for col in df_hourly.columns:
    s = df_hourly[col]
    
    # teste de Tuckey  
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_mask = (s < lower) | (s > upper)

    # % de outliers
    outlier_count = outlier_mask.sum()
    total = len(s)
    outlier_pct = (outlier_count / total * 100) if total > 0 else 0.0

    # insere na lista
    outlier_summary.append({
        "column": col,
        "percent_outliers": outlier_pct
    })

outlier_summary_df = (
    pd.DataFrame(outlier_summary)
      .sort_values("percent_outliers", ascending=False)
      .reset_index(drop=True)
)

# plotando
plt.figure(figsize=(12, 5))
plt.bar(outlier_summary_df["column"], outlier_summary_df["percent_outliers"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Outliers (%)")
plt.xlabel("Variáveis")
plt.title("Percentagem de Outliers por Variáveis")
plt.tight_layout()
plt.savefig("graphs/Outliers.png", dpi=300, bbox_inches="tight")
plt.show()

 #Tendencia, sazonalidade etc

# preparando os dados
y = df_hourly[TARGET]
fit = MSTL(y, periods=(24, 168)).fit()
trend = pd.Series(fit.trend, index=y.index)

# definindo as varáveis
seas_day  = fit.seasonal.iloc[:, 0].rename('seasonal_24h')
seas_week = fit.seasonal.iloc[:, 1].rename('seasonal_168h')
resid = pd.Series(fit.resid, index=y.index)

# plotando
fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
axes[0].plot(y, color='tab:blue');      axes[0].set_ylabel('Observed')
axes[1].plot(trend, color='tab:orange');axes[1].set_ylabel('Trend')
axes[2].plot(seas_day, color='tab:green');axes[2].set_ylabel('Daily\nSeasonality')
axes[3].plot(seas_week, color='tab:green');axes[3].set_ylabel('Weekly\nSeasonality')
axes[4].plot(resid, color='tab:red');   axes[4].axhline(0, lw=1); axes[4].set_ylabel('Residuals')
for ax in axes: ax.grid(alpha=0.3)
fig.suptitle(f'STL/MSTL Decomposition of {TARGET}', fontsize=14)
plt.tight_layout()
plt.savefig("graphs/MSTL.png", dpi=300, bbox_inches="tight")
plt.show()

# calculando a força das tendências
Fs_day  = 1 - resid.var(ddof=0) / (resid.add(seas_day, fill_value=0).var(ddof=0))
Fs_week = 1 - resid.var(ddof=0) / (resid.add(seas_week, fill_value=0).var(ddof=0))
Ft      = 1 - resid.var(ddof=0) / (resid.add(trend,    fill_value=0).var(ddof=0))

# printando
print({'trend_strength': round(Ft,3), 'daily_strength': round(Fs_day,3), 'weekly_strength': round(Fs_week,3)})

# ACF

# TARGET observável
obs = pd.Series(fit.observed, index=y.index)

# # de lags
max_lags = 200

# plotando
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# observável
plot_acf(obs.dropna(), lags=max_lags, ax=axes[0], zero=False)
axes[0].set_title("ACF — Observed % Silica Concentrate")

# residual
plot_acf(resid.dropna(), lags=max_lags, ax=axes[1], zero=False)
axes[1].set_title("ACF — Residuals (after trend & seasonality)")

for ax in axes: 
    for lag in (24, 168): 
            ax.axvline(lag, linestyle='--', alpha=0.4)
    
plt.tight_layout()
plt.savefig("graphs/ACF.png", dpi=300, bbox_inches="tight")
plt.show()

#ADFtest

series = y.dropna()

result = adfuller(series, autolag='AIC')

print(f"ADF Statistic: {result[0]:.4f}")
print(f"p-value:       {result[1]:.4f}")
print("Critical Values:")
for key, value in result[4].items():
    print(f"   {key}: {value:.4f}")

# Correlation

corr = df_hourly.corr()
plt.figure(figsize=(20,20))
sns.heatmap(corr, annot=True)
plt.title('Heatmap da correlação')
plt.savefig("graphs/Correlação.png", dpi=300, bbox_inches="tight")
plt.show()

target_corr = corr[TARGET].drop(labels=[TARGET])
top10_abs_idx = target_corr.abs().sort_values(ascending=False).head(5).index
top10_by_abs = target_corr.loc[top10_abs_idx]

print(f"Top 5 features most correlated with {TARGET} (by absolute correlation):")
print(top10_by_abs)

# PCA

cols.drop([TARGET])
X = df_hourly[cols]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=5) 
principal_components = pca.fit_transform(X_scaled)


explained_var = np.cumsum(pca.explained_variance_ratio_) * 100
plt.figure(figsize=(8,4))
plt.plot(range(1, len(explained_var)+1), explained_var, marker='o')
plt.title('Cumulative Explained Variance by Principal Components')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance (%)')
plt.grid(True)
plt.savefig("graphs/PCA1.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(principal_components[:,0], principal_components[:,1],
            c=df_hourly.loc[X.index, TARGET], 
            cmap='coolwarm', alpha=0.6)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA: First Two Principal Components (colored by % Silica Concentrate)')
plt.colorbar(label='% Silica')
plt.savefig("graphs/PCA2.png", dpi=300, bbox_inches="tight")
plt.show()

loadings = pd.DataFrame(pca.components_.T,
                        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
                        index=cols)
print("Top contributing variables to PC1:")
print(loadings['PC1'].abs().sort_values(ascending=False).head(10))

from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Prepare data: Drop the target and any non-numeric or redundant columns
X = df_hourly.drop(columns=[TARGET])
y = df_hourly[TARGET]

# Initialize and fit the model
# We use a shallow depth for importance to avoid overfitting to noise
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X, y)

# Extract and sort importance
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)

# Plotting
plt.figure(figsize=(10, 8))
importances.plot(kind='barh', color='teal')
plt.title('Feature Importance: Drivers of % Silica Concentrate')
plt.xlabel('Importance Score')
plt.grid(axis='x', alpha=0.3)
plt.show()