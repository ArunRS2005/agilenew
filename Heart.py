{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "f87339c9-273c-4da8-9b06-efa5f1daa3a7",
   "metadata": {},
   "source": [
    "Naive Bayes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "8d3b151a-38f2-4c37-b871-c28496a1625a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0 1 1 0 1 0 1 1 0 1 0 1 0 1 1 1 1 0 0 1 0 1 1 1 0 1 0 1 1 0 1 0 1 1 0 1 0\n",
      " 0 0 0 1 1 1 1 1 1 1 1 1 0 0 1 1 1 0 0 1 1 1 0 1 1 1 0 1 1 0 1 1 0 1 1 1 0\n",
      " 1 1 1 1 1 0 1 1 0 1 0 1 0 1 1 0 1 1 1 0 1 0 1 1 1 1 0 1 1 1 1 0 1 1 1 0 1\n",
      " 0 0 1 0 0 1 0 0 1 0 0 1 1 1 1 1 1 1 0 0 1 0 0 1 0 1 0 0 1 0 0 0 1 0 1 0 0\n",
      " 1 1 1 1 0 0 1 1 1 0 1 1 1 1 1 1 1 0 1 1 1 0 1 0 0 1 0 1 1 1 1 0 0 1 0 1 0\n",
      " 0 0 1 1 0 0 1 1 1 1 1 1 1 0 0 1 1 0 1 1 1 0 0 1 0 1 1 0 1 1 1 1 1 1 1 0 0\n",
      " 1 0 0 0 0 0 1 0 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 0 0 1 0 1 1 1 1 0 1]\n",
      "74.31906614785993\n"
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
    "df = pd.read_csv(\"heart.csv\")\n",
    "\n",
    "x = df.drop('sex', axis=1)\n",
    "y = df['sex']\n",
    "\n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)\n",
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
   "id": "2f802708-4985-4407-998e-bfb050d54ea2",
   "metadata": {},
   "source": [
    "decision tree"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "9dc85df7-4be7-4f41-bc96-bb0f745a0316",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "98.83268482490273"
      ]
     },
     "execution_count": 19,
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
    "df = pd.read_csv(\"heart.csv\")\n",
    "df.head()\n",
    "x=df.drop('sex',axis=1)\n",
    "y=df['sex']\n",
    "x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)\n",
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
   "id": "a3332d1c-3345-4601-b217-41b9db8b5621",
   "metadata": {},
   "source": [
    "random forest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "a23fd171-313a-454e-8ab0-403ab480b97b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0 1 1 0 1 0 1 1 1 1 0 1 1 1 1 0 1 1 1 1 0 1 1 1 0 1 0 1 0 0 1 0 1 1 1 1 0\n",
      " 1 0 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 0 0 0 1 1 1 1 1 0 1 1 1 0 1 1 0 1 0 1 1\n",
      " 0 1 1 1 1 1 1 1 0 1 1 1 0 1 1 0 1 1 1 0 1 1 1 1 0 1 1 1 1 1 1 0 1 1 1 0 1\n",
      " 0 0 0 1 0 1 1 0 1 0 1 1 1 1 1 1 1 1 0 0 0 0 1 1 0 1 1 0 1 1 0 0 1 0 1 1 0\n",
      " 0 1 1 1 1 1 0 1 0 0 1 1 1 0 1 1 1 0 1 0 1 0 1 1 0 0 0 1 1 1 1 0 1 1 1 1 0\n",
      " 0 1 1 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 1 1 1 0 1 1 1 0 0 1 1 1 1 0\n",
      " 1 0 1 1 1 0 1 0 1 1 1 0 1 1 1 0 0 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1 1 1 0]\n",
      "97.66536964980544\n"
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
    "df = pd.read_csv(\"heart.csv\")\n",
    "x = df.drop('sex', axis=1)\n",
    "y = df['sex']\n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)\n",
    "\n",
    "model = RandomForestClassifier(n_estimators=100, random_state=42)\n",
    "model.fit(x_train, y_train)\n",
    "\n",
    "y_pred = model.predict(x_test)\n",
    "print(y_pred)\n",
    "print(accuracy_score(y_test, y_pred) * 100)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d046aaa2-d3c8-439b-ae36-a05df91334ba",
   "metadata": {},
   "source": [
    "SVM"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "ad73ed6d-c1c2-4a7e-be4c-bfe9b53de220",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predictions: [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1\n",
      " 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 0 1 1\n",
      " 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1\n",
      " 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 0 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1\n",
      " 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 0 1 1 0 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1\n",
      " 1 1 1 1 0 0 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0\n",
      " 1 1 0 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1\n",
      " 1 1 0 0 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 0 1 1 1 1 1 0 1 1 1 1 1 1 1 0\n",
      " 1 1 1 1 1 1 1 1 1 1 1 1]\n",
      "Accuracy: 72.40259740259741\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.metrics import accuracy_score\n",
    "df = pd.read_csv(\"heart.csv\")\n",
    "x = df.drop('sex', axis=1)\n",
    "y = df['sex']\n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)\n",
    "classifier = SVC(kernel='linear', random_state=0)\n",
    "classifier.fit(x_train, y_train)\n",
    "y_pred = classifier.predict(x_test)\n",
    "print(\"Predictions:\", y_pred)\n",
    "print(\"Accuracy:\", accuracy_score(y_test, y_pred) * 100)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5a021edd-4643-4b5d-86c3-570c5fcf945d",
   "metadata": {},
   "source": [
    "KNN"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "8f543ca7-6909-4e08-9bc6-1c0c40e6748d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predictions: [0 1 1 1 1 0 0 1 1 1 1 0 0 1 1 1 1 1 0 0 1 1 0 1 1 1 0 1 0 0 1 1 1 1 1 1 1\n",
      " 1 1 1 1 1 1 0 0 1 1 1 1 1 1 1 1 1 0 1 0 0 1 1 1 1 0 1 1 1 0 1 1 1 1 0 1 1\n",
      " 0 1 0 1 1 0 1 1 0 1 1 1 1 1 1 0 1 0 0 0 0 1 1 1 1 1 0 1 1 1 1 1 0 0 1 1 1\n",
      " 1 0 0 1 0 1 1 0 1 0 1 1 1 1 1 0 1 1 0 1 1 0 0 0 1 1 1 1 1 1 1 1 0 1 1 1 0\n",
      " 1 1 1 1 1 1 0 1 0 0 1 1 1 1 1 1 1 1 0 1 1 0 1 1 0 1 1 1 1 1 1 0 1 1 1 1 1\n",
      " 0 1 1 1 0 0 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 1 1 0 1 1 1 0 1 0 1 0 1 0\n",
      " 1 0 0 1 1 0 0 0 0 1 1 0 1 0 0 0 0 1 1 0 1 1 1 0 1 0 1 1 1 1 1 1 1 1 0]\n",
      "Accuracy: 75.09727626459144\n"
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
    "df = pd.read_csv(\"heart.csv\")\n",
    "x = df.drop('sex', axis=1)  \n",
    "y = df['sex']           \n",
    "\n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)\n",
    "knn_classifier = KNeighborsClassifier(n_neighbors=5)\n",
    "knn_classifier.fit(x_train, y_train)\n",
    "y_pred = knn_classifier.predict(x_test)\n",
    "print(\"Predictions:\", y_pred)\n",
    "print(\"Accuracy:\", accuracy_score(y_test, y_pred) * 100)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "97bdda20-fee3-413d-8570-99089e7a8423",
   "metadata": {},
   "source": [
    "Logistic Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "92d45660-eca3-428a-b6e8-f730f29e0244",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predictions: [1 0 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 1 0 1 1 1 0 1 1 1 1 1 1 1 1 0 1 1 0\n",
      " 0 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1\n",
      " 1 1 1 1 1 0 1 1 0 1 0 1 0 1 1 1 1 1 1 0 1 1 1 1 1 1 0 1 1 0 1 1 1 1 1 1 1\n",
      " 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 0 1 1 1 0 1 1 1 1 1 1 0 1 0 0 1 1 1 0\n",
      " 1 1 1 1 1 0 1 1 1 0 1 1 1 1 1 1 1 0 1 1 0 0 1 1 0 1 1 0 1 1 1 1 1 1 1 1 1\n",
      " 1 1 1 1 0 0 1 1 1 1 1 1 0 1 1 1 0 1 1 1 1 0 1 0 1 0 1 1 1 1 1 1 1 1 1 1 0\n",
      " 1 1 0 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 0 0 1 1 1 1 1 1 1 1]\n",
      "Accuracy: 71.59533073929961\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import accuracy_score\n",
    "df = pd.read_csv(\"heart.csv\")\n",
    "x = df.drop('sex', axis=1)\n",
    "y = df['sex']\n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)\n",
    "classifier = LogisticRegression(max_iter=1000)\n",
    "classifier.fit(x_train, y_train)\n",
    "y_pred = classifier.predict(x_test)\n",
    "print(\"Predictions:\", y_pred)\n",
    "print(\"Accuracy:\", accuracy_score(y_test, y_pred) * 100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "a4a66e8c-037d-4ea1-a9ca-fab6d0c34589",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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PXr0wNq1azFmzBjExsZi0KBBePXqFWrUqIGDBw/C0tJS2Wbu3LkwMjJCx44dERsbi0aNGmHt2rUwNDTM9OdDRJRTTD//XN8l6M1YlwL6LoHSKMuss6NPXGeHiChtGHYoK8j26+wQERERpQeGHSIiIlK1LLuoIBFRZsmph2R4OIZyCo7sEBERkaox7BAREZGqMewQERGRqjHsEBERkaox7BAREZGqMewQERGRqjHsEBERkaox7BAREZGqMewQERGRqjHsEBERkaox7BAREZGqMewQERGRqjHsEBERkaox7BAREZGqMewQERGRqjHsEBERkaox7BAREZGqMewQERGRqjHsEBERkaox7BAREZGqMewQERGRqjHsEBERkaox7BAREZGqMewQERGRqjHsEBERkaox7BAREZGqMewQERGRqjHsEBERkaox7BAREZGqMewQERGRqjHsEBERkaox7BAREZGqMewQERGRqjHsEBERkaox7BAREZGqMewQERGRqjHsEBERkaox7BAREZGqMewQERGRqjHsEBERkaox7BAREZGqMewQERGRqjHsEBERkaox7BAREZGqMewQERGRqjHsEBERkaox7BAREZGqMewQERGRqjHsEBERkaox7BAREZGqMewQERGRqjHsEBERkaox7BAREZGqMewQERGRqjHsEBERkaox7BAREZGqMewQERGRqjHsEBERkapl6bCTmJiICRMmwMnJCWZmZihWrBimTJmC5ORkpY+IYNKkSbC3t4eZmRnq16+Pq1ev6rFqIiIiykqydNiZMWMGli1bhkWLFuHatWuYOXMmZs2ahYULFyp9Zs6ciTlz5mDRokUIDg6GnZ0dmjRpgujoaD1WTkRERFlFlg47p06dQuvWrdGyZUsULVoU7du3R9OmTXHmzBkA70Z15s2bh/Hjx6Ndu3YoX7481q1bhzdv3sDf31/P1RMREVFWkKXDTu3atXH48GHcvHkTAHDx4kWcOHECLVq0AADcvXsXYWFhaNq0qbKNqakp6tWrh5MnT37wcePi4hAVFaXzQ0REROpkpO8C/omXlxciIyNRpkwZGBoaIikpCdOmTcNXX30FAAgLCwMA2Nra6mxna2uL+/fvf/Bx/fz8MHny5IwrnIiIiLKMLD2ys3XrVmzcuBH+/v44d+4c1q1bh9mzZ2PdunU6/TQajc5tEUnVlpK3tzciIyOVnwcPHmRI/URERKR/WXpkx9PTE2PHjkXnzp0BABUqVMD9+/fh5+eHHj16wM7ODsC7EZ5ChQop24WHh6ca7UnJ1NQUpqamGVs8ERERZQlZemTnzZs3MDDQLdHQ0FA59dzJyQl2dnY4dOiQcn98fDwCAwPh6uqaqbUSERFR1pSlR3Y8PDwwbdo0fPbZZyhXrhzOnz+POXPmoHfv3gDeHb4aMWIEfH19UbJkSZQsWRK+vr4wNzdHly5d9Fw9ERERZQVZOuwsXLgQEydOxKBBgxAeHg57e3sMGDAA3333ndJnzJgxiI2NxaBBg/Dq1SvUqFEDBw8ehKWlpR4rJyIioqxCIyKi7yL0LSoqCnny5EFkZCSsrKz0XQ4RZbLp55/ruwS9GOtS4JO3zan7DPhv+43S18d+f2fpOTtERERE/xXDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqVqWvjaWGnBJdSIiIv3iyA4RERGpGsMOERERqRoPYxEREWWSnDq1Qd/TGjiyQ0RERKrGsENERESqxrBDREREqsawQ0RERKrGsENERESqxrBDREREqsZTz4lUhKe1EhGlxpEdIiIiUjWGHSIiIlI1HsaiLImHY4iIKL1wZIeIiIhUjWGHiIiIVI1hh4iIiFSNYYeIiIhUjWGHiIiIVI1hh4iIiFSNYYeIiIhUjWGHiIiIVI1hh4iIiFSNYYeIiIhUjWGHiIiIVI1hh4iIiFSNYYeIiIhUjWGHiIiIVI1hh4iIiFSNYYeIiIhUjWGHiIiIVI1hh4iIiFSNYYeIiIhUjWGHiIiIVI1hh4iIiFSNYYeIiIhUjWGHiIiIVI1hh4iIiFSNYYeIiIhUzSgtnUUEgYGBOH78OO7du4c3b96gYMGCcHFxQePGjeHg4JBRdRIRERF9ko8a2YmNjYWvry8cHBzQvHlz7Nu3DxERETA0NMTt27fh4+MDJycntGjRAkFBQRldMxEREdFH+6iRnVKlSqFGjRpYtmwZ3NzcYGxsnKrP/fv34e/vj06dOmHChAno169fuhdLRERElFYfFXb279+P8uXL/2MfR0dHeHt749tvv8X9+/fTpTgiIiKi/+qjDmP9W9BJycTEBCVLlvzkgoiIiIjSU5omKKeUmJiI5cuX49ixY0hKSkKtWrUwePBg5MqVKz3rIyIiIvpPPjnsDBs2DDdv3kS7du2QkJCA9evX48yZM9i8eXN61kdERET0n3x02AkICEDbtm2V2wcPHsSNGzdgaGgIAHBzc8MXX3yR/hUSERER/QcfvajgqlWr0KZNGzx69AgAUKVKFXzzzTf49ddfsWfPHowZMwbVq1fPsEKJiIiIPsVHh529e/eic+fOqF+/PhYuXIgVK1bAysoK48ePx8SJE+Hg4AB/f/+MrJWIiIgozdI0Z6dz585o1qwZPD094ebmhuXLl+OHH37IqNqIiIiI/rM0Xxsrb968WLlyJWbNmoVu3brB09MTsbGxGVEbAODRo0f4+uuvYW1tDXNzc1SuXBlnz55V7hcRTJo0Cfb29jAzM0P9+vVx9erVDKuHiIiIspePDjsPHjxAp06dUKFCBXTt2hUlS5bE2bNnYWZmhsqVK2P//v3pXtyrV69Qq1YtGBsbY//+/QgJCcEPP/yAvHnzKn1mzpyJOXPmYNGiRQgODoadnR2aNGmC6OjodK+HiIiIsp+PDjvdu3eHRqPBrFmzYGNjgwEDBsDExARTpkzBrl274Ofnh44dO6ZrcTNmzICDgwPWrFmDzz//HEWLFkWjRo1QvHhxAO9GdebNm4fx48ejXbt2KF++PNatW4c3b95w/hAREREBSEPYOXPmDKZNm4ZmzZphzpw5uHTpknJf2bJl8fvvv6Nx48bpWtzu3btRrVo1dOjQATY2NnBxccHKlSuV++/evYuwsDA0bdpUaTM1NUW9evVw8uTJDz5uXFwcoqKidH6IiIhInT467FSpUgXfffcdDh48CC8vL1SoUCFVn/79+6drcX/99ReWLl2KkiVL4sCBA/jmm28wbNgwrF+/HgAQFhYGALC1tdXZztbWVrnvffz8/JAnTx7lx8HBIV3rJiIioqzjo8PO+vXrERcXh5EjR+LRo0dYvnx5RtYFAEhOTkaVKlXg6+sLFxcXDBgwAP369cPSpUt1+mk0Gp3bIpKqLSVvb29ERkYqPw8ePMiQ+omIiEj/PvrUc0dHR/z0008ZWUsqhQoVgrOzs05b2bJlsWPHDgCAnZ0dgHcjPIUKFVL6hIeHpxrtScnU1BSmpqYZUDERERFlNR81shMTE5OmB01r/w+pVasWbty4odN28+ZNODo6AgCcnJxgZ2eHQ4cOKffHx8cjMDAQrq6u6VIDERERZW8fFXZKlCgBX19fPH78+IN9RASHDh1C8+bNsWDBgnQpbuTIkQgKCoKvry9u374Nf39/rFixAoMHDwbw7vDViBEj4Ovri4CAAFy5cgU9e/aEubk5unTpki41EBERUfb2UYexjh07hgkTJmDy5MmoXLkyqlWrBnt7e+TKlQuvXr1CSEgITp06BWNjY3h7e6fbROXq1asjICAA3t7emDJlCpycnDBv3jx07dpV6TNmzBjExsZi0KBBePXqFWrUqIGDBw/C0tIyXWogIiKi7O2jwk7p0qWxfft2PHz4ENu3b8fvv/+OkydPIjY2FgUKFFBOCW/RogUMDNK8KPM/cnd3h7u7+wfv12g0mDRpEiZNmpSuv5eIiIjUIU3XxipSpAhGjhyJkSNHZlQ9REREROkqfYdhiIiIiLIYhh0iIiJSNYYdIiIiUjWGHSIiIlI1hh0iIiJStTSHnaJFi2LKlCkIDQ3NiHqIiIiI0lWaw863336Ln3/+GcWKFUOTJk2wZcsWxMXFZURtRERERP9ZmsPO0KFDcfbsWZw9exbOzs4YNmwYChUqhCFDhuDcuXMZUSMRERHRJ/vkOTuVKlXC/Pnz8ejRI/j4+ODHH39E9erVUalSJaxevRoikp51EhEREX2SNK2gnFJCQgICAgKwZs0aHDp0CF988QX69OmDx48fY/z48fjtt9/g7++fnrUSERERpVmaw865c+ewZs0abN68GYaGhujWrRvmzp2LMmXKKH2aNm2KunXrpmuhRERERJ8izWGnevXqaNKkCZYuXYo2bdrA2Ng4VR9nZ2d07tw5XQokIiIi+i/SHHb++usvODo6/mMfCwsLrFmz5pOLIiIiIkovaZ6gHB4ejtOnT6dqP336NM6cOZMuRRERERGllzSHncGDB+PBgwep2h89eoTBgwenS1FERERE6SXNYSckJARVqlRJ1e7i4oKQkJB0KYqIiIgovaQ57JiamuLp06ep2p88eQIjo08+k52IiIgoQ6Q57DRp0gTe3t6IjIxU2iIiIjBu3Dg0adIkXYsjIiIi+q/SPBTzww8/oG7dunB0dISLiwsA4MKFC7C1tcWGDRvSvUAiIiKi/yLNYadw4cK4dOkSNm3ahIsXL8LMzAy9evXCV1999d41d4iIiIj06ZMm2VhYWKB///7pXQsRERFRuvvkGcUhISEIDQ1FfHy8TnurVq3+c1FERERE6eWTVlBu27YtLl++DI1Go1zdXKPRAACSkpLSt0IiIiKi/yDNZ2MNHz4cTk5OePr0KczNzXH16lX8/vvvqFatGo4dO5YBJRIRERF9ujSP7Jw6dQpHjhxBwYIFYWBgAAMDA9SuXRt+fn4YNmwYzp8/nxF1EhEREX2SNI/sJCUlIXfu3ACAAgUK4PHjxwAAR0dH3LhxI32rIyIiIvqP0jyyU758eVy6dAnFihVDjRo1MHPmTJiYmGDFihUoVqxYRtRIRERE9MnSHHYmTJiAmJgYAMD3338Pd3d31KlTB9bW1ti6dWu6F0hERET0X6Q57Li5uSn/X6xYMYSEhODly5fIly+fckYWERERUVaRpjk7iYmJMDIywpUrV3Ta8+fPz6BDREREWVKawo6RkREcHR25lg4RERFlG2k+G2vChAnw9vbGy5cvM6IeIiIionSV5jk7CxYswO3bt2Fvbw9HR0dYWFjo3H/u3Ll0K46IiIjov0pz2GnTpk0GlEFERESUMdIcdnx8fDKiDiIiIqIMkeY5O0RERETZSZpHdgwMDP7xNHOeqUVERERZSZrDTkBAgM7thIQEnD9/HuvWrcPkyZPTrTAiIiKi9JDmsNO6detUbe3bt0e5cuWwdetW9OnTJ10KIyIiIkoP6TZnp0aNGvjtt9/S6+GIiIiI0kW6hJ3Y2FgsXLgQRYoUSY+HIyIiIko3aT6M9fcLfooIoqOjYW5ujo0bN6ZrcURERET/VZrDzty5c3XCjoGBAQoWLIgaNWogX7586VocERER0X+V5rDTs2fPDCiDiIiIKGOkec7OmjVrsH379lTt27dvx7p169KlKCIiIqL0kuawM336dBQoUCBVu42NDXx9fdOlKCIiIqL0kuawc//+fTg5OaVqd3R0RGhoaLoURURERJRe0hx2bGxscOnSpVTtFy9ehLW1dboURURERJRe0hx2OnfujGHDhuHo0aNISkpCUlISjhw5guHDh6Nz584ZUSMRERHRJ0vz2Vjff/897t+/j0aNGsHI6N3mycnJ6N69O+fsEBERUZaT5rBjYmKCrVu34vvvv8eFCxdgZmaGChUqwNHRMSPqIyIiIvpP0hx2tEqWLImSJUumZy1ERERE6S7Nc3bat2+P6dOnp2qfNWsWOnTokC5FEREREaWXNIedwMBAtGzZMlV7s2bN8Pvvv6dLUURERETpJc1h5/Xr1zAxMUnVbmxsjKioqHQpioiIiCi9pDnslC9fHlu3bk3VvmXLFjg7O6dLUURERETpJc0TlCdOnIgvv/wSd+7cQcOGDQEAhw8fxubNm997zSwiIiIifUpz2GnVqhV27doFX19f/PTTTzAzM0PFihXx22+/oV69ehlRIxEREdEn+6RTz1u2bPneScoXLlxA5cqV/2tNREREROkmzXN2/i4yMhJLlixBlSpVULVq1fSo6YP8/Pyg0WgwYsQIpU1EMGnSJNjb28PMzAz169fH1atXM7QOIiIiyj4+OewcOXIEXbt2RaFChbBw4UK0aNECZ86cSc/adAQHB2PFihWoWLGiTvvMmTMxZ84cLFq0CMHBwbCzs0OTJk0QHR2dYbUQERFR9pGmsPPw4UN8//33KFasGL766ivkz58fCQkJ2LFjB77//nu4uLhkSJGvX79G165dsXLlSuTLl09pFxHMmzcP48ePR7t27VC+fHmsW7cOb968gb+/f4bUQkRERNnLR4edFi1awNnZGSEhIVi4cCEeP36MhQsXZmRtisGDB6Nly5Zo3LixTvvdu3cRFhaGpk2bKm2mpqaoV68eTp48+cHHi4uLQ1RUlM4PERERqdNHT1A+ePAghg0bhoEDB2bqNbG2bNmCc+fOITg4ONV9YWFhAABbW1uddltbW9y/f/+Dj+nn54fJkyenb6FERESUJX30yM7x48cRHR2NatWqoUaNGli0aBGePXuWkbXhwYMHGD58ODZu3IhcuXJ9sJ9Go9G5LSKp2lLy9vZGZGSk8vPgwYN0q5mIiIiylo8OOzVr1sTKlSvx5MkTDBgwAFu2bEHhwoWRnJyMQ4cOZciE4LNnzyI8PBxVq1aFkZERjIyMEBgYiAULFsDIyEgZ0dGO8GiFh4enGu1JydTUFFZWVjo/REREpE5pPhvL3NwcvXv3xokTJ3D58mV8++23mD59OmxsbNCqVat0La5Ro0a4fPkyLly4oPxUq1YNXbt2xYULF1CsWDHY2dnh0KFDyjbx8fEIDAyEq6trutZCRERE2dN/WmendOnSmDlzJh4+fIjNmzenV00KS0tLlC9fXufHwsIC1tbWKF++vLLmjq+vLwICAnDlyhX07NkT5ubm6NKlS7rXQ0RERNnPJ62g/HeGhoZo06YN2rRpkx4PlyZjxoxBbGwsBg0ahFevXqFGjRo4ePAgLC0tM70WIiIiynrSJexkpmPHjunc1mg0mDRpEiZNmqSXeoiIiChr+8+XiyAiIiLKyhh2iIiISNUYdoiIiEjVGHaIiIhI1Rh2iIiISNUYdoiIiEjVGHaIiIhI1Rh2iIiISNUYdoiIiEjVGHaIiIhI1Rh2iIiISNUYdoiIiEjVGHaIiIhI1Rh2iIiISNUYdoiIiEjVGHaIiIhI1Rh2iIiISNUYdoiIiEjVGHaIiIhI1Rh2iIiISNUYdoiIiEjVGHaIiIhI1Rh2iIiISNUYdoiIiEjVGHaIiIhI1Rh2iIiISNUYdoiIiEjVGHaIiIhI1Rh2iIiISNUYdoiIiEjVGHaIiIhI1Rh2iIiISNUYdoiIiEjVGHaIiIhI1Rh2iIiISNUYdoiIiEjVGHaIiIhI1Rh2iIiISNUYdoiIiEjVGHaIiIhI1Rh2iIiISNUYdoiIiEjVGHaIiIhI1Rh2iIiISNUYdoiIiEjVGHaIiIhI1Rh2iIiISNUYdoiIiEjVGHaIiIhI1Rh2iIiISNUYdoiIiEjVGHaIiIhI1Rh2iIiISNUYdoiIiEjVGHaIiIhI1Rh2iIiISNUYdoiIiEjVGHaIiIhI1Rh2iIiISNUYdoiIiEjVsnTY8fPzQ/Xq1WFpaQkbGxu0adMGN27c0OkjIpg0aRLs7e1hZmaG+vXr4+rVq3qqmIiIiLKaLB12AgMDMXjwYAQFBeHQoUNITExE06ZNERMTo/SZOXMm5syZg0WLFiE4OBh2dnZo0qQJoqOj9Vg5ERERZRVG+i7gn/z66686t9esWQMbGxucPXsWdevWhYhg3rx5GD9+PNq1awcAWLduHWxtbeHv748BAwboo2wiIiLKQrL0yM7fRUZGAgDy588PALh79y7CwsLQtGlTpY+pqSnq1auHkydPfvBx4uLiEBUVpfNDRERE6pRtwo6IYNSoUahduzbKly8PAAgLCwMA2Nra6vS1tbVV7nsfPz8/5MmTR/lxcHDIuMKJiIhIr7JN2BkyZAguXbqEzZs3p7pPo9Ho3BaRVG0peXt7IzIyUvl58OBButdLREREWUOWnrOjNXToUOzevRu///47ihQporTb2dkBeDfCU6hQIaU9PDw81WhPSqampjA1Nc24gomIiCjLyNIjOyKCIUOGYOfOnThy5AicnJx07ndycoKdnR0OHTqktMXHxyMwMBCurq6ZXS4RERFlQVl6ZGfw4MHw9/fHzz//DEtLS2UeTp48eWBmZgaNRoMRI0bA19cXJUuWRMmSJeHr6wtzc3N06dJFz9UTERFRVpClw87SpUsBAPXr19dpX7NmDXr27AkAGDNmDGJjYzFo0CC8evUKNWrUwMGDB2FpaZnJ1RIREVFWlKXDjoj8ax+NRoNJkyZh0qRJGV8QERERZTtZes4OERER0X/FsENERESqxrBDREREqsawQ0RERKrGsENERESqxrBDREREqsawQ0RERKrGsENERESqxrBDREREqsawQ0RERKrGsENERESqxrBDREREqsawQ0RERKrGsENERESqxrBDREREqsawQ0RERKrGsENERESqxrBDREREqsawQ0RERKrGsENERESqxrBDREREqsawQ0RERKrGsENERESqxrBDREREqsawQ0RERKrGsENERESqxrBDREREqsawQ0RERKrGsENERESqxrBDREREqsawQ0RERKrGsENERESqxrBDREREqsawQ0RERKrGsENERESqxrBDREREqsawQ0RERKrGsENERESqxrBDREREqsawQ0RERKrGsENERESqxrBDREREqsawQ0RERKrGsENERESqxrBDREREqsawQ0RERKrGsENERESqxrBDREREqsawQ0RERKrGsENERESqxrBDREREqsawQ0RERKrGsENERESqxrBDREREqsawQ0RERKrGsENERESqxrBDREREqsawQ0RERKrGsENERESqxrBDREREqqaasLNkyRI4OTkhV65cqFq1Ko4fP67vkoiIiCgLUEXY2bp1K0aMGIHx48fj/PnzqFOnDpo3b47Q0FB9l0ZERER6poqwM2fOHPTp0wd9+/ZF2bJlMW/ePDg4OGDp0qX6Lo2IiIj0LNuHnfj4eJw9exZNmzbVaW/atClOnjypp6qIiIgoqzDSdwH/1fPnz5GUlARbW1uddltbW4SFhb13m7i4OMTFxSm3IyMjAQBRUVHpXt/b19Hp/pjZRVSUySdvm1P323/ZZwD326fifku7nLrPAO63T/Ff36Mfftx339si8o/9sn3Y0dJoNDq3RSRVm5afnx8mT56cqt3BwSFDasupUu9h+jfcZ5+G++3TcL99Gu63tMvofRYdHY08efJ88P5sH3YKFCgAQ0PDVKM44eHhqUZ7tLy9vTFq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",
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
    "accuracies = [75.09727626459144, 71.59533073929961,  72.40259740259741, 97.66536964980544, 98.83268482490273, 74.31906614785993]\n",
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
   "id": "edaf635f-3864-4572-ba2a-57fde1c95a8d",
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
