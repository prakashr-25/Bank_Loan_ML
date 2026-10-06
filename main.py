import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import(
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score, silhouette_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from xgboost import XGBClassifier

#Load Data
df = pd.read_csv("data/loan_train.csv")

#Feature Engineering
df["application_date"] = pd.to_datetime(df["application_date"])
df["application_year"] = df["application_date"].dt.year
df["application_month"] = df["application_date"].dt.month
df["income_loan_ratio"] = df["annual_income"] / df["loan_amount"]
df["log_annual_income"] = np.log1p(df["annual_income"])

df = df.drop(columns="application_date")
X = df.drop(columns="loan_status")
y = df["loan_status"]

num = [
    "loan_amount", "interest_rate", "annual_income", "dti", "credit_score",
    "months_since_last_delinq", "application_year", "application_month",
    "income_loan_ratio", "log_annual_income"
]

cat = ["term", "employment_length", "home_ownership", "loan_purpose", "verification_status"]

ord_col = ["loan_grade"]

def make_preprocessor(numeric=num):
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat),
        ("ord", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(categories=[["A","B","C","D","E","F","G"]]))
        ]), ord_col)
    ])

def classifier(model):
    return Pipeline([("preprocessor", make_preprocessor()), ("model", model)])

#Task-1
print("\nTASK-1")
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

print("Train:", len(X_train), "Validation:", len(X_val), "Test:", len(X_test))

depths = [1, 3, 5, 10, None]
train_scores, val_scores = [], []
for depth in depths:
    m = classifier(DecisionTreeClassifier(max_depth=depth, random_state=42))
    m.fit(X_train, y_train)
    tr = accuracy_score(y_train, m.predict(X_train))
    va = accuracy_score(y_val, m.predict(X_val))
    train_scores.append(tr); val_scores.append(va)

    print("Depth", depth, ": train", round(tr, 2), "val", round(va, 2))

plt.plot(["1","3","5","10","None"], train_scores, marker="o", label="Train")
plt.plot(["1","3","5","10","None"], val_scores, marker="o", label="Validation")
plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree Overfitting")
plt.legend()
plt.grid(True)
plt.savefig("outputs/task1_overfitting.png")
plt.close()

logistic = classifier(LogisticRegression(max_iter=1000))
cv = cross_val_score(logistic, X_train, y_train, cv=5, scoring="accuracy")

print("\n5-Fold Cross Validation:")
print("CV:", round(cv.mean(), 2), "+/-", round(cv.std(), 2))
leak_X = df[["loan_amount", "annual_income", "dti", "credit_score"]]
wrong_scaled = StandardScaler().fit_transform(leak_X)
a, b, c, d = train_test_split(wrong_scaled, y, test_size=0.20, random_state=42, stratify=y)
wrong = LogisticRegression().fit(a, c)
wrong_acc = accuracy_score(d, wrong.predict(b))
a, b, c, d = train_test_split(leak_X, y, test_size=0.20, random_state=42, stratify=y)
scaler = StandardScaler()
a = scaler.fit_transform(a); b = scaler.transform(b)
correct = LogisticRegression().fit(a, c)
correct_acc = accuracy_score(d, correct.predict(b))

print("Wrong Scaling:", round(wrong_acc, 2))
print("Correct Scaling:", round(correct_acc, 2))
print("Leakage: fit preprocessing only on training data; Pipeline helps prevent leakage.")
print("Spam Detection: Supervised")
print("Customer Segmentation: Unsupervised")
print("House Price Prediction: Supervised")
print("Anomaly Detection: Unsupervised")
print("Workflow: Collect -> Clean -> Split -> Train -> Evaluate -> Improve -> Deploy")


#Task-2
print("\nTASK-2")
KNNImputer(n_neighbors=5).fit_transform(X_train[num])
print("KNNImputer completed")
print("Pipeline CV Accuracy:", round(cv.mean(), 3))

logistic.fit(X_train, y_train)
joblib.dump(logistic, "outputs/loan_default_pipeline.joblib")
loaded_model = joblib.load("outputs/loan_default_pipeline.joblib")

new_rows = pd.DataFrame({
    "loan_amount":[8000,22000,12000],
    "term":["36 months","60 months","36 months"],
    "interest_rate":[8.5,19.5,12.0],
    "loan_grade":["B","F","C"],
    "employment_length":["5 years","2 years","10+ years"],
    "home_ownership":["OWN","RENT","MORTGAGE"],
    "annual_income":[85000,38000,72000],
    "application_year":[2026,2026,2026],
    "application_month":[1,2,3],
    "loan_purpose":["car","debt_consolidation","home_improvement"],
    "dti":[10.5,31.0,17.0],
    "credit_score":[755,570,690],
    "months_since_last_delinq":[np.nan,12,36],
    "verification_status":["Verified","Source Verified","Not Verified"],
    "income_loan_ratio":[85000/8000,38000/22000,72000/12000],
    "log_annual_income":np.log1p([85000,38000,72000])
})

print("Predictions for 3 new rows:", loaded_model.predict(new_rows))


#Task-3
print("\nTASK-3")
tree = classifier(DecisionTreeClassifier(max_depth=5, random_state=42))
forest = classifier(RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42))
tree.fit(X_train, y_train); forest.fit(X_train, y_train)
print("Random Forest OOB:", round(forest.named_steps["model"].oob_score_, 3))

xgb_pre = make_preprocessor()
xgb_pre.fit(X_train, y_train)
X_val_xgb = xgb_pre.transform(X_val)
xgb = classifier(XGBClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=4,
    eval_metric="logloss", early_stopping_rounds=20, random_state=42))
xgb.fit(X_train, y_train, model__eval_set=[(X_val_xgb, y_val)], model__verbose=False)

print("Logistic Probabilities:\n", np.round(logistic.predict_proba(X_val.head(5)), 3))
print("Predictions:", logistic.predict(X_val.head(5)))

models = {"Logistic":logistic, "Decision Tree":tree, "Random Forest":forest, "XGBoost":xgb}
for name in ["Random Forest", "XGBoost"]:
    m = models[name]
    names = m.named_steps["preprocessor"].get_feature_names_out()
    imp = m.named_steps["model"].feature_importances_
    table = pd.DataFrame({"Feature":names, "Importance":imp})

    print("\nTop 5", name, "Features:")
    print(table.sort_values("Importance", ascending=False).head(5))

print("\nValidation Results")
for name, model in models.items():
    print(name, round(accuracy_score(y_val, model.predict(X_val)), 3))

X_reg = df.drop(columns=["loan_status", "loan_amount", "income_loan_ratio"])
y_reg = df["loan_amount"]
X_reg_train, X_reg_val = X_reg.loc[X_train.index], X_reg.loc[X_val.index]
y_reg_train, y_reg_val = y_reg.loc[X_train.index], y_reg.loc[X_val.index]
reg_num = ["interest_rate","annual_income","dti","credit_score","months_since_last_delinq",
           "application_year","application_month","log_annual_income"]

def reg_model(model):
    return Pipeline([("preprocessor", make_preprocessor(reg_num)), ("model", model)])
linear = reg_model(LinearRegression())
ridge = reg_model(Ridge(alpha=1))
lasso = reg_model(Lasso(alpha=100, max_iter=10000))

for m in [linear, ridge, lasso]:
    m.fit(X_reg_train, y_reg_train)

coef = pd.DataFrame({
    "Feature":linear.named_steps["preprocessor"].get_feature_names_out(),
    "Linear":linear.named_steps["model"].coef_,
    "Ridge":ridge.named_steps["model"].coef_,
    "Lasso":lasso.named_steps["model"].coef_
})

print("\nCoefficient Comparison:\n", coef.head(10))
print("\nLasso Zero Coefficients:", np.sum(lasso.named_steps["model"].coef_ == 0))
cluster = df[["annual_income","loan_amount","dti"]]
cluster_scaled = StandardScaler().fit_transform(cluster)
inertia, sil = [], []

for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(cluster_scaled)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(cluster_scaled, labels))

    print("k =", k, "Silhouette:", round(sil[-1], 3))

plt.plot(range(2,9), inertia, marker="o")
plt.xlabel("k"); plt.ylabel("Inertia")
plt.title("Elbow Curve")
plt.savefig("outputs/task3_kmeans_elbow.png")
plt.close()
plt.plot(range(2,9), sil, marker="o")
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score")
plt.savefig("outputs/task3_kmeans_silhouette.png")
plt.close()

best_k = list(range(2,9))[np.argmax(sil)]
km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
segments = cluster.copy()
segments["Cluster"] = km.fit_predict(cluster_scaled)

print("\nCustomer Segments:\n", segments.groupby("Cluster").mean().round(2))


#Task-4
print("\nTASK-4")
probs = {}

for name, model in models.items():
    pred = model.predict(X_val)
    prob = model.predict_proba(X_val)[:,1]
    probs[name] = prob

    print("", name)
    print("Accuracy:", round(accuracy_score(y_val,pred),3))
    print("Precision:", round(precision_score(y_val,pred,zero_division=0),3))
    print("Recall:", round(recall_score(y_val,pred),3))
    print("F1:", round(f1_score(y_val,pred),3))
    print("AUC:", round(roc_auc_score(y_val,prob),3))
    print("Confusion Matrix:\n", confusion_matrix(y_val,pred))


for name in ["Logistic","Random Forest","XGBoost"]:
    fpr, tpr, _ = roc_curve(y_val, probs[name])
    auc = roc_auc_score(y_val, probs[name])
    plt.plot(fpr, tpr, label=name+" "+str(round(auc,3)))
plt.plot([0,1],[0,1],linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.savefig("outputs/task4_roc_curves.png")
plt.close()

best_threshold, best_precision, best_recall = 0.5, 0, 0

for threshold in np.arange(0.05,0.96,0.01):
    pred = (probs["XGBoost"] >= threshold).astype(int)
    rec = recall_score(y_val,pred)
    pre = precision_score(y_val,pred,zero_division=0)

    if rec >= 0.75 and pre > best_precision:
        best_threshold, best_precision, best_recall = threshold, pre, rec

print("Threshold:", round(best_threshold,2),
      "Recall:", round(best_recall,3),
      "Precision:", round(best_precision,3))

rf_grid = GridSearchCV(
    classifier(RandomForestClassifier(random_state=42)),
    {"model__n_estimators":[50,100],
     "model__max_depth":[3,5,None],
     "model__min_samples_split":[2,5]},
    cv=3, scoring="roc_auc")
rf_grid.fit(X_train,y_train)

print("RF Best:", rf_grid.best_params_, round(rf_grid.best_score_,3))

xgb_random = RandomizedSearchCV(
    classifier(XGBClassifier(eval_metric="logloss",random_state=42)),
    {"model__n_estimators":[50,100,150,200],
     "model__max_depth":[2,3,4,5],
     "model__learning_rate":[0.01,0.05,0.1,0.2]},
    n_iter=15, cv=3, scoring="roc_auc", random_state=42)
xgb_random.fit(X_train,y_train)

print("XGB Best:", xgb_random.best_params_, round(xgb_random.best_score_,3))
print("\nRegression Results")

for name, model in [("Linear",linear),("Ridge",ridge),("Lasso",lasso)]:
    pred = model.predict(X_reg_val)

    print(
        name,
        "MAE:",round(mean_absolute_error(y_reg_val,pred),2),
        "RMSE:",round(np.sqrt(mean_squared_error(y_reg_val,pred)),2),
        "R2:",round(r2_score(y_reg_val,pred),3)
    )
    
baseline = np.full(len(y_reg_val), y_reg_train.mean())

print(
    "Baseline",
    "MAE:",round(mean_absolute_error(y_reg_val,baseline),2),
    "RMSE:",round(np.sqrt(mean_squared_error(y_reg_val,baseline)),2),
    "R2:",round(r2_score(y_reg_val,baseline),3)
)

candidates = {
    **models,
    "Tuned RF": rf_grid.best_estimator_,
    "Tuned XGB": xgb_random.best_estimator_
}

val_auc = {
    name: roc_auc_score(
        y_val,
        model.predict_proba(X_val)[:, 1]
    )
    for name, model in candidates.items()
}

best_name = max(val_auc, key=val_auc.get)
best_model = candidates[best_name]

test_pred = best_model.predict(X_test)
test_prob = best_model.predict_proba(X_test)[:, 1]

final_auc = roc_auc_score(y_test, test_prob)
final_f1 = f1_score(y_test, test_pred)


reg_models = {
    "Linear": linear,
    "Ridge": ridge,
    "Lasso": lasso
}

best_reg_name = ""
best_reg_r2 = -999

for name, model in reg_models.items():

    pred = model.predict(X_reg_val)
    score = r2_score(y_reg_val, pred)

    if score > best_reg_r2:
        best_reg_r2 = score
        best_reg_name = name
        best_reg_model = model


reg_pred = best_reg_model.predict(X_reg_val)
reg_mae = mean_absolute_error(y_reg_val, reg_pred)
reg_rmse = np.sqrt(
    mean_squared_error(y_reg_val, reg_pred)
)
reg_r2 = r2_score(y_reg_val, reg_pred)


baseline = np.full(
    len(y_reg_val),
    y_reg_train.mean()
)
base_mae = mean_absolute_error(
    y_reg_val,
    baseline
)
base_rmse = np.sqrt(
    mean_squared_error(y_reg_val, baseline)
)
base_r2 = r2_score(
    y_reg_val,
    baseline
)


print("\nFINAL RESULT")

print(
    "Best model:", best_name,
    "| ROC-AUC", round(final_auc, 2),
    "| F1", round(final_f1, 2)
)

print(
    "Regression:", best_reg_name,
    "| MAE", round(reg_mae, 2),
    "| RMSE", round(reg_rmse, 2),
    "| R2", round(reg_r2, 2)
)

print(
    "Baseline:",
    "MAE", round(base_mae, 2),
    "| RMSE", round(base_rmse, 2),
    "| R2", round(base_r2, 2)
)