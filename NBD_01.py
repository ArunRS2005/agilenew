{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "95b97a61-e2c0-4ab6-b938-2cd7244f06b1",
   "metadata": {},
   "source": [
    "## Naive Bayes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "7adee3e4-17cf-406b-8a5b-134e2f8df2a4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1 1 1 0 0 0 1 0 0 1 0 0 0 0 0 1 1 1 1 0 0 1 1 0 0 0 1 1 0 0 0 1 0 1 0 1 1\n",
      " 0 0 0 1 0 0 0 0 0 1 1 1 1 0 1 0 1 1 1 1 1 1 1 0 1 1 1 0 1 1 0 1 1 0 0 0 0\n",
      " 1 1 0 0 1 1 0 0 0 0 0 0 0 1 0 1 0 0 1 0 1 1 0 1 1 1 1 1 1 0 0 1 1 0 1 0 1\n",
      " 0 1 0 0 1 0 0 1 1 1 0 0 0 0 1 1 0 1 1 1 0 0 0 1 1 1 1 0 1 1 0 1 1 1 1 0 0\n",
      " 1 1 0 1 0 0 1 1 1 0 0 1 1 1 0 1 1 1 0 0 1 1 1 0 1 0 0 1 0 1 1 1 1 1 0 0 0\n",
      " 1 1 1 0 1 1 1 1 0 0 1 0 0 0 0 1 1 1 1 0 1 1 1 0 0 1 1 1 1 1 1 1 1 0 0 1 1\n",
      " 0 0 1 1 1 1 0 0 1 0 0 1 0 1 1 0 1 0 1 0 1 0 1 0 0 0 0 0 1 0 1 0 0 1 0 1 0\n",
      " 0 1 1 0 0 0 0 0 1 0 0 0 1 0 1 0 0 0 1 1 1 1 0 0 1 1 1 0 1 0 1 1 1 1 0 1 0\n",
      " 1 1 0]\n",
      "93.31103678929766\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "from sklearn.metrics import accuracy_score\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "df = pd.read_csv(\"NBD.csv\")\n",
    "\n",
    "x = df.drop('diabetes', axis=1)\n",
    "y = df['diabetes']\n",
    "\n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)\n",
    "\n",
    "model = GaussianNB()\n",
    "model.fit(x_train, y_train)\n",
    "\n",
    "y_pred = model.predict(x_test)\n",
    "print(y_pred)\n",
    "predict_score=accuracy_score(y_test,y_pred)*100\n",
    "print(predict_score)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "305d5219-5dc9-48fd-92c0-b3d30f832eec",
   "metadata": {},
   "source": [
    "## Decision tree"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "9932e5e2-2b14-4769-b332-31f734cdaf2f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "93.64548494983278"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.metrics import accuracy_score\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "df = pd.read_csv(\"NBD.csv\")\n",
    "df.head()\n",
    "x=df.drop('diabetes',axis=1)\n",
    "y=df['diabetes']\n",
    "x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)\n",
    "model=DecisionTreeClassifier()\n",
    "model.fit(x_train,y_train)\n",
    "y_pred = model.predict(x_test)\n",
    "y_pred\n",
    "score=accuracy_score(y_test,y_pred)\n",
    "score*100"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7685f180-d095-4159-92c4-717a27f4ae31",
   "metadata": {},
   "source": [
    "## Random Forest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "0ce1e611-6d2b-42f6-9262-5a4779658133",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predictions: [1 1 1 0 0 0 1 0 0 1 0 0 0 0 0 1 1 1 1 0 0 1 1 0 0 0 1 1 0 0 0 1 0 1 0 1 1\n",
      " 0 0 0 1 0 0 0 0 0 1 1 1 1 0 1 0 1 1 1 1 1 1 1 0 1 1 1 0 1 0 0 1 1 0 0 0 0\n",
      " 1 1 0 0 1 1 0 0 0 0 0 0 0 1 0 1 0 0 1 0 1 1 0 1 1 1 1 1 1 0 0 1 1 0 1 0 1\n",
      " 0 0 0 0 1 0 0 1 1 1 0 0 0 0 1 1 0 1 1 1 0 0 0 1 1 1 1 0 1 1 0 1 1 1 1 0 0\n",
      " 1 1 0 1 0 0 1 1 1 0 0 1 1 1 1 1 1 1 0 0 1 1 1 0 1 0 0 1 0 1 1 1 1 1 0 0 0\n",
      " 1 1 1 0 1 1 1 1 0 0 1 0 0 0 0 1 1 1 1 0 1 1 1 0 0 1 1 1 1 1 1 1 1 0 0 1 1\n",
      " 0 0 1 1 1 1 0 0 1 0 0 1 0 1 1 1 1 0 1 0 1 0 1 0 0 0 0 0 1 0 0 0 0 1 0 1 0\n",
      " 0 1 1 0 0 0 0 0 1 0 0 0 1 0 1 0 0 0 1 1 1 0 0 0 1 1 1 0 1 0 1 1 1 1 0 1 0\n",
      " 1 1 0]\n",
      "Accuracy: 93.9799331103679\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import accuracy_score\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "\n",
    "df = pd.read_csv(\"NBD.csv\")\n",
    "x = df.drop('diabetes', axis=1)\n",
    "y = df['diabetes']\n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)\n",
    "\n",
    "model = RandomForestClassifier(n_estimators=100, random_state=42)\n",
    "model.fit(x_train, y_train)\n",
    "\n",
    "y_pred = model.predict(x_test)\n",
    "print(\"Predictions:\", y_pred)\n",
    "print(\"Accuracy:\", accuracy_score(y_test, y_pred) * 100)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fd3cbf6c-b506-49a3-aabd-cb7315f85521",
   "metadata": {},
   "source": [
    "## SVM"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "12cc1c33-1192-4d1d-9637-abcfa6d05c10",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predictions: [1 1 1 0 0 0 1 0 0 1 0 0 0 0 0 1 1 1 1 0 0 1 1 0 0 0 1 1 0 0 0 1 0 1 0 1 1\n",
      " 0 0 0 1 0 0 0 0 0 1 1 1 1 0 1 0 1 1 1 1 1 1 0 0 1 1 0 0 1 0 0 1 1 0 0 0 0\n",
      " 1 1 0 0 1 1 0 0 0 0 0 0 0 1 0 1 0 0 1 0 1 1 0 1 1 1 1 1 1 0 0 1 1 0 1 0 1\n",
      " 0 0 0 0 1 0 0 1 1 1 0 0 0 0 1 1 0 1 1 1 0 0 0 1 1 1 1 0 1 1 0 1 1 1 1 0 0\n",
      " 1 1 0 1 0 0 1 1 1 0 0 1 1 1 0 1 1 1 0 0 1 1 1 0 1 0 0 1 0 1 1 1 1 1 0 0 0\n",
      " 1 1 1 0 1 1 1 1 0 0 1 0 0 0 0 1 1 1 1 0 1 1 0 0 0 1 1 1 1 1 1 1 1 0 0 1 1\n",
      " 0 0 1 1 1 1 0 0 1 0 0 1 0 1 1 1 1 0 1 0 1 0 1 0 0 0 0 0 1 0 0 0 0 1 0 1 0\n",
      " 0 1 1 0 0 0 0 0 1 0 0 0 1 0 1 0 0 0 1 1 1 0 0 0 1 1 1 0 1 0 1 1 1 1 0 1 0\n",
      " 1 1 0]\n",
      "Accuracy: 92.64214046822742\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.metrics import accuracy_score\n",
    "df = pd.read_csv(\"NBD.csv\")\n",
    "x = df.drop('diabetes', axis=1)\n",
    "y = df['diabetes']\n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)\n",
    "classifier = SVC(kernel='linear', random_state=0)\n",
    "classifier.fit(x_train, y_train)\n",
    "y_pred = classifier.predict(x_test)\n",
    "print(\"Predictions:\", y_pred)\n",
    "print(\"Accuracy:\", accuracy_score(y_test, y_pred) * 100)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4372e646-dab3-4f51-9aa8-2a24edbdc168",
   "metadata": {},
   "source": [
    "## KNN"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "0db6e1cc-6112-448b-a235-c6b6cfa3bb70",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predictions: [1 1 1 0 0 0 1 0 0 1 0 0 0 0 0 1 1 1 1 0 0 1 1 0 0 0 1 1 0 0 0 1 0 1 0 1 1\n",
      " 0 0 0 1 0 0 0 0 0 1 1 1 1 0 1 0 1 1 1 1 1 1 1 0 1 1 1 0 1 0 0 1 1 0 0 0 0\n",
      " 1 1 0 0 1 1 0 0 0 0 0 0 0 1 0 1 0 1 1 0 1 1 0 1 1 1 1 1 1 0 0 1 1 0 1 0 1\n",
      " 0 0 0 0 1 0 0 1 1 1 0 0 0 0 1 1 0 1 1 1 0 0 0 1 1 1 1 0 1 1 0 1 1 1 1 0 0\n",
      " 1 1 0 1 0 0 1 1 1 0 0 1 1 1 0 1 1 1 0 0 1 1 1 0 1 0 0 1 0 1 1 1 1 1 0 0 0\n",
      " 1 1 1 0 1 1 1 1 0 0 1 0 0 0 0 1 1 1 1 0 1 1 1 0 0 1 1 1 1 1 1 1 1 0 0 1 1\n",
      " 0 0 1 1 1 1 1 0 1 0 1 1 0 1 1 1 1 0 1 0 1 0 1 0 0 0 1 0 1 0 0 0 0 1 0 1 0\n",
      " 0 1 1 0 0 0 0 0 1 0 0 0 1 0 1 0 0 0 1 1 1 1 0 0 1 1 1 0 1 0 1 1 1 1 0 1 0\n",
      " 1 1 0]\n",
      "Accuracy: 93.9799331103679\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.metrics import accuracy_score\n",
    "\n",
    "df = pd.read_csv(\"NBD.csv\")\n",
    "x = df.drop('diabetes', axis=1)  \n",
    "y = df['diabetes']           \n",
    "\n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)\n",
    "knn_classifier = KNeighborsClassifier(n_neighbors=5)\n",
    "knn_classifier.fit(x_train, y_train)\n",
    "y_pred = knn_classifier.predict(x_test)\n",
    "print(\"Predictions:\", y_pred)\n",
    "print(\"Accuracy:\", accuracy_score(y_test, y_pred) * 100)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "97695b8d-dcac-4b57-911f-194f411495fc",
   "metadata": {},
   "source": [
    "## Logistic Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "7d12bb5d-41da-413f-8529-4b084f87ac98",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predictions: [1 1 1 0 0 0 1 0 0 1 0 0 0 0 0 1 1 1 1 0 0 1 1 0 0 0 1 1 0 0 0 1 0 1 0 1 1\n",
      " 0 0 0 1 0 0 0 0 0 1 1 1 1 0 1 0 1 1 1 1 1 1 0 0 1 1 0 0 1 0 0 1 1 0 0 0 0\n",
      " 1 1 0 0 1 1 0 0 0 0 0 0 0 1 0 1 0 0 1 0 1 1 0 1 1 1 1 1 1 0 0 1 1 0 1 0 1\n",
      " 0 0 0 0 1 0 0 1 1 1 0 0 0 0 1 1 0 1 1 1 0 0 0 1 1 1 1 0 1 1 0 1 1 1 1 0 0\n",
      " 1 1 0 1 0 0 1 1 1 0 0 1 1 1 0 1 1 1 0 0 1 1 1 0 1 0 0 1 0 1 1 1 1 1 0 0 0\n",
      " 1 1 1 0 1 1 1 1 0 0 1 0 0 0 0 1 1 1 1 0 1 1 0 0 0 1 1 1 1 1 1 1 1 0 0 1 1\n",
      " 0 0 1 1 1 1 0 0 1 0 0 1 0 1 1 1 1 0 1 0 1 0 1 0 0 0 0 0 1 0 0 0 0 1 0 1 0\n",
      " 0 1 1 0 0 0 0 0 1 0 0 0 1 0 1 0 0 0 1 1 1 0 0 0 1 1 1 0 1 0 1 1 1 1 0 1 0\n",
      " 1 1 0]\n",
      "Accuracy: 92.64214046822742\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import accuracy_score\n",
    "df = pd.read_csv(\"NBD.csv\")\n",
    "x = df.drop('diabetes', axis=1)\n",
    "y = df['diabetes']\n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)\n",
    "classifier = LogisticRegression(max_iter=1000)\n",
    "classifier.fit(x_train, y_train)\n",
    "y_pred = classifier.predict(x_test)\n",
    "print(\"Predictions:\", y_pred)\n",
    "print(\"Accuracy:\", accuracy_score(y_test, y_pred) * 100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "7181add7-9ef3-42d7-89a3-2d6316748878",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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HyIiIvp8ZZkwIyIYOnQoPDw8ULp0aQBAZGQkAMDOzk5nWzs7O+W6vwsKCkLOnDmVH0dHx09bOBEREelVlgkzAwYMwPnz57F27do012k0Gp3LIpKmTcvf3x8vX75Ufu7evftJ6iUiIqKsQa9jZrQGDhyIrVu34tChQyhYsKDSbm9vD+BtD03+/PmV9sePH6fprdEyNTWFqanppy2YiIiIsgy99syICAYMGIDNmzdj3759cHFx0bnexcUF9vb22LNnj9KWkJCAgwcPwt3dPbPLJSIioixIrz0z/fv3R3BwMH799VdYWVkp42By5swJc3NzaDQaDB48GIGBgShatCiKFi2KwMBAWFhYoEOHDvosnYiIiLIIvYaZ+fPnAwBq1aql075s2TJ07doVADBs2DDExcWhX79+ePHiBapUqYLdu3fDysoqk6slIiKirEivYUZE/nUbjUaDMWPGYMyYMZ++ICIiIlKdLDObiYiIiOhDMMwQERGRqjHMEBERkaoxzBAREZGqMcwQERGRqjHMEBERkaoxzBAREZGqMcwQERGRqjHMEBERkaoxzBAREZGqMcwQERGRqjHMEBERkaoxzBAREZGqMcwQERGRqjHMEBERkaoxzBAREZGqMcwQERGRqjHMEBERkaoxzBAREZGqMcwQERGRqjHMEBERkaoxzBAREZGqMcwQERGRqjHMEBERkaoxzBAREZGqMcwQERGRqjHMEBERkaoxzBAREZGqMcwQERGRqjHMEBERkaoxzBAREZGqMcwQERGRqjHMEBERkaoxzBAREZGqMcwQERGRqjHMEBERkaoxzBAREZGqMcwQERGRqjHMEBERkaoxzBAREZGqMcwQERGRqjHMEBERkaoxzBAREZGqMcwQERGRqjHMEBERkaoxzBAREZGqMcwQERGRqjHMEBERkaoxzBAREZGqMcwQERGRqjHMEBERkaoxzBAREZGqMcwQERGRqjHMEBERkaoxzBAREZGqMcwQERGRqjHMEBERkaoxzBAREZGqMcwQERGRqjHMEBERkaoxzBAREZGqMcwQERGRqjHMEBERkaqpIszMmzcPLi4uMDMzQ8WKFXH48GF9l0RERERZRJYPM+vXr8fgwYMxcuRInDlzBtWrV0ejRo0QERGh79KIiIgoC8jyYWbatGno0aMHevbsiZIlS2LGjBlwdHTE/Pnz9V0aERER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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "accuracies = [93.9799331103679, 92.64214046822742,  92.64214046822742, 93.9799331103679, 93.64548494983278, 93.31103678929766]\n",
    "models = ['KNN', 'Logistic Regression', 'SVM', 'Random Forest', 'Decision Tree', 'Gaussian NB']\n",
    "plt.bar(models, accuracies, color='skyblue')\n",
    "plt.xlabel('Model')\n",
    "plt.ylabel('Accuracy (%)')\n",
    "plt.title('Model Accuracy Comparison')\n",
    "plt.xticks(rotation=45)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e4d08c19-0802-4324-bfae-45e5b908c399",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
