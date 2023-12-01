# %%
from lightgbm import LGBMClassifier
from sklearn.datasets import make_moons

# %%
model = LGBMClassifier(boosting_type='gbdt', num_leaves=31,
                       max_depth=- 1, learning_rate=0.1, n_estimators=300, device="gpu")

train, label = make_moons(n_samples=300000, shuffle=True,
                          noise=0.3, random_state=None)

model.fit(train, label)

# %%
