import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


# 1. LOAD FORM DATA
df = pd.read_csv("student_data_700.csv")

print("Columns:", df.columns.tolist())


# 2. CLEAN & PARSE DATA

def parse_range(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    try:
        if "-" in x:
            low, high = x.split("-")
            return (float(low) + float(high)) / 2
        return float(x)
    except:
        return np.nan


df["SCR"] = df["SCR"].apply(parse_range)

for col in ["CGPA", "SLP", "ATT"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# 3. FEATURES & TARGET
X = df[["SLP", "ATT", "SCR"]].to_numpy(dtype=float)
y = df["CGPA"].to_numpy(dtype=float)


# 4. DROP MISSING VALUES
mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
X = X[mask]
y = y[mask]


# 5. STANDARDIZE FEATURES
scaler = StandardScaler()
X_std = scaler.fit_transform(X)


# 6. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X_std, y, test_size=0.2, random_state=42
)


# 7. TRAIN MODEL
model = LinearRegression()
model.fit(X_train, y_train)


# 8. EVALUATION
y_pred = model.predict(X_test)

print("R²:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))


# 9. 3D VISUALIZATION
# Conditional regression plane (SCR fixed at mean)

SCR_mean = X_std[:, 2].mean()

x1_surf, x2_surf = np.meshgrid(
    np.linspace(X_std[:, 0].min(), X_std[:, 0].max(), 50),
    np.linspace(X_std[:, 1].min(), X_std[:, 1].max(), 50)
)

y_surf = (
    model.intercept_
    + model.coef_[0] * x1_surf
    + model.coef_[1] * x2_surf
    + model.coef_[2] * SCR_mean
)

y_plane_pred = (
    model.intercept_
    + model.coef_[0] * X_std[:, 0]
    + model.coef_[1] * X_std[:, 1]
    + model.coef_[2] * SCR_mean
)

above_plane = y >= y_plane_pred
below_plane = ~above_plane


# PLOT
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    X_std[above_plane, 0],
    X_std[above_plane, 1],
    y[above_plane],
    s=90,
    alpha=0.85,
    label="Above plane"
)

ax.scatter(
    X_std[below_plane, 0],
    X_std[below_plane, 1],
    y[below_plane],
    s=60,
    alpha=0.4,
    label="Below plane"
)

ax.plot_surface(
    x1_surf,
    x2_surf,
    y_surf,
    alpha=0.35
)

ax.set_xlabel("Sleep (std)")
ax.set_ylabel("Attendance (std)")
ax.set_zlabel("CGPA")

ax.set_title("Conditional Regression Plane (SCR fixed at mean)")
ax.legend()

ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.show()


# 10. DATA SUMMARY
print("Total responses:", df.shape[0])
print("Usable responses:", len(y))
print(df.count())
