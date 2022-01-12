# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
import numpy as np


def label_function(val, data):
    return f'{val / 100 * len(data):.0f}\n{val:.0f}%'

def plot_pie_charts(axs, column):
    data.loc[data['Biopsy'] == 1].groupby(column).size().plot(kind='pie',
                                                                autopct=lambda pct: label_function(pct, data.loc[
                                                                    data['Biopsy'] == 1]),
                                                                ax=axs[0],
                                                                title=column+" if Biopsy is 1",
                                                                labels=['0', '1'])
    axs[0].set_ylabel('')
    data.loc[data['Biopsy'] == 0].groupby(column).size().plot(kind='pie', autopct=lambda pct: label_function(pct,
                                                                                                               data.loc[
                                                                                                                   data[
                                                                                                                       'Biopsy'] == 0]),
                                                                ax=axs[1],
                                                                title=column+" if Biopsy is 0",
                                                                labels=['0', '1'])
    axs[1].set_ylabel('')
    data.loc[data[column] == 1.0].groupby('Biopsy').size().plot(kind='pie', autopct=lambda pct: label_function(pct,
                                                                                                                 data.loc[
                                                                                                                     data[
                                                                                                                         column] == 1]),
                                                                  ax=axs[2],
                                                                  title="Biopsy if "+column+" is 1",
                                                                  labels=['0', '1'])
    axs[2].set_ylabel('')
    data.loc[data[column] == 0].groupby('Biopsy').size().plot(kind='pie', autopct=lambda pct: label_function(pct,
                                                                                                               data.loc[
                                                                                                                   data[
                                                                                                                       column] == 0]),
                                                                ax=axs[3],
                                                                title="Biopsy if "+column+" is 0",
                                                                labels=['0', '1'])
    axs[3].set_ylabel('')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Data loading
    fileName = "kag_risk_factors_cervical_cancer.csv"
    filePath = os.getcwd() + '/' + fileName

    data = pd.read_csv(filePath)

    # Class balance
    fig1, ax1 = plt.subplots()
    data.groupby('Biopsy').size().plot(kind='pie', autopct=lambda pct: label_function(pct, data), ax=ax1,
                                       title="Balansiranost klasa",
                                       labels=['negative', 'positive'])
    ax1.set_ylabel('')

    # Replace to NaN and drop sparse columns
    data.replace("?", np.nan, inplace=True)
    print(data.isna().sum())
    perc = 20.0
    min_count = int(((100 - perc) / 100) * data.shape[0] + 1)
    data.dropna(axis=1, thresh=min_count, inplace=True)
    data = data.astype(float)

    fig2, axs2 = plt.subplots(nrows=2, ncols=2)
    ax2 = axs2.ravel()
    plot_pie_charts(ax2, 'Smokes')
    fig3, axs3 = plt.subplots(nrows=2, ncols=2)
    ax3 = axs3.ravel()
    plot_pie_charts(ax3, 'Hormonal Contraceptives')
    fig4, axs4 = plt.subplots(nrows=2, ncols=2)
    ax4 = axs4.ravel()
    plot_pie_charts(ax4, 'IUD')
    fig5, axs5 = plt.subplots(nrows=2, ncols=2)
    ax5 = axs5.ravel()
    plot_pie_charts(ax5, 'STDs')
    fig6, axs6 = plt.subplots(nrows=2, ncols=2)
    ax6 = axs6.ravel()
    plot_pie_charts(ax6, 'Dx:Cancer')
    fig7, axs7 = plt.subplots(nrows=2, ncols=2)
    ax7 = axs7.ravel()
    plot_pie_charts(ax7, 'Dx:CIN')
    fig8, axs8 = plt.subplots(nrows=2, ncols=2)
    ax8 = axs8.ravel()
    plot_pie_charts(ax8, 'Dx:HPV')
    fig9, axs9 = plt.subplots(nrows=2, ncols=2)
    ax9 = axs9.ravel()
    plot_pie_charts(ax9, 'Dx')
    fig10, axs10 = plt.subplots(nrows=2, ncols=2)
    ax10 = axs10.ravel()
    plot_pie_charts(ax10, 'Hinselmann')
    fig11, axs11 = plt.subplots(nrows=2, ncols=2)
    ax11 = axs11.ravel()
    plot_pie_charts(ax11, 'Schiller')
    fig12, axs12 = plt.subplots(nrows=2, ncols=2)
    ax12 = axs12.ravel()
    plot_pie_charts(ax12, 'Citology')

    labels = ['Age', 'Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies', 'STDs (number)']
    positive_means = data.loc[data['Biopsy'] == 1][data.columns.intersection(labels)].mean()
    negative_means = data.loc[data['Biopsy'] == 0][data.columns.intersection(labels)].mean()
    positive_stds = data.loc[data['Biopsy'] == 1][data.columns.intersection(labels)].std()
    negative_stds = data.loc[data['Biopsy'] == 0][data.columns.intersection(labels)].std()

    fig13, ax13 = plt.subplots()
    X_axis = np.arange(len(labels))
    ax13.bar(X_axis - 0.2, positive_means, 0.4, yerr=positive_stds, label='Biopsy 1', ecolor='black', capsize=6)
    ax13.bar(X_axis + 0.2, negative_means, 0.4, yerr=negative_stds, label='Biopsy 0', ecolor='black', capsize=6)
    plt.xticks(X_axis, labels)
    ax13.set_xlabel("Obelezja")
    ax13.set_ylabel("Srednja vrednost")
    ax13.set_title("Srednja vrednost prediktora")
    ax13.legend()

    # Handling missing values
    imputer = KNNImputer(n_neighbors=10)
    imputed = imputer.fit_transform(data)
    data_imputed = pd.DataFrame(imputed, columns=data.columns)

    fig14, ax14 = plt.subplots()
    ax14.imshow(data_imputed.corr())
    print(data_imputed.corr().iloc[:, -1])

    data.drop(axis=1, index=data_imputed.corr() == np.nan, inplace=True)
    # Split
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

    # Handling missing values
    imputer = KNNImputer(n_neighbors=10)
    imputed = imputer.fit_transform(X_train)
    X_train_imputed = pd.DataFrame(imputed, columns=X_train.columns)


    print(1)
    plt.show()
