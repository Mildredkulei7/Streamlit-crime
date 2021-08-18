{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Streamlit-crime.ipynb",
      "provenance": [],
      "authorship_tag": "ABX9TyNIG+7vLnO18dVUTAs1E9Eu",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Mildredkulei7/Streamlit-crime/blob/main/Streamlit_crime.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "yt_roZ1PrqFl"
      },
      "source": [
        "import streamlit as st\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import sklearn\n",
        "\n",
        "#Create the horizontal sections of our app\n",
        "header = st.container()\n",
        "desc = st.container()\n",
        "plot = st.container()\n",
        "prediction = st.container()\n",
        "model = st.container()\n",
        "\n",
        "#The first section\n",
        "with header:\n",
        "    st.title('Crime Analysis')\n",
        "    st.write(\"\"\"\n",
        "    # Text Classification\n",
        "    \"\"\")\n",
        "    st.write('This is an application that classifies crime related text into various subcategories')\n",
        "    col1, col2 = st.columns(2)\n",
        "\n",
        "with model:\n",
        "    st.title('Classification Algorithms')\n",
        "    models = st.sidebar.selectbox('Select Classification Model', ('Naive Bayes','KNN'))\n",
        "    dataset = st.sidebar.selectbox('Select dataset', (\"crime-vectorised.csv\"))\n",
        "    st.write('Classification model : ',models)\n",
        "    st.write('Dataset : ',dataset)\n",
        "\n",
        "#Loading the dataset\n",
        "    def get_data(dataset_name):\n",
        "        data = pd.read_csv('/content/crime-vectorised.csv')\n",
        "        X = data['clean_text']\n",
        "        y = data['label']\n",
        "        return X,y\n",
        "    X,y = get_data(dataset)\n",
        "\n",
        "    st.write('Number of tweets : ', len(X))\n",
        "    st.write('Dataset classes : ', y.unique().tolist())\n",
        "\n",
        "#Adding model parameters\n",
        "    def model_params(clf_name):\n",
        "        params = dict()\n",
        "        if models == 'KNN':\n",
        "            from sklearn.neighbors import KNeighborsClassifier\n",
        "            cl_gs=KNeighborsClassifier(n_neighbors=9)\n",
        "            cl_gs.fit(X_train_tfidf,y_train)\n",
        "            k = st.sidebar.slider('Number of K-folds', 1,10)\n",
        "            params['K'] = k\n",
        "        elif models == 'Naive Bayes':\n",
        "            from sklearn.naive_bayes import MultinomialNB  \n",
        "            mn= MultinomialNB().fit(X_train_tfidf, y_train)\n",
        "            params['alpha'] = 1.0\n",
        "        return params\n",
        "\n",
        "    model_params(models)\n",
        "\n",
        "#Second section of the app\n",
        "with desc:\n",
        "    st.title('Overview')\n",
        "    st.write('This is the overview of the app')\n",
        "\n",
        "#The third section\n",
        "#with plot:\n",
        "\n",
        "\n",
        "#The input and prediction section\n",
        "#with prediction:"
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}