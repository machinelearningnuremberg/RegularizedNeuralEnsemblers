#load sklearn example and train 30 models from scikit learn
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.utils import all_estimators
from sklearn.datasets import load_diabetes
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from regularized_neural_ensemblers.model import NeuralEnsembler
from regularized_neural_ensemblers.trainer import Trainer
from regularized_neural_ensemblers.trainer_args import TrainerArgs


def get_base_functions():

    X, y = load_diabetes(return_X_y=True)
    # 2. Split dataset
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.75, random_state=1)  # 0.25 x 0.8 = 0.2

    # 3. Get 30 classifiers from sklearn
    all_regressors = [cls for name, cls in all_estimators(type_filter='regressor')]
    selected_regressors = all_regressors[:30]  # Limit to first 30

    # 4. Initialize prediction matrix
    model_names = []
    predictions_val = []
    predictions_test = []

    # 5. Train models and predict on validation set
    for idx, cls in enumerate(selected_regressors):
        try:
            model = cls()
            model.fit(X_train, y_train)

            pred = model.predict(X_val)
            predictions_val.append(pred[:, np.newaxis, np.newaxis])

            pred_test = model.predict(X_test)
            predictions_test.append(pred_test[:, np.newaxis, np.newaxis])
            model_names.append(cls.__name__)

        except Exception as e:
            print(f"Model {cls.__name__} failed: {e}")

    base_functions_val = np.concatenate(predictions_val, axis=-1)
    base_functions_val[np.isnan(base_functions_val)] = 1/base_functions_val.shape[1]

    base_functions_test = np.concatenate(predictions_test, axis=-1)
    base_functions_test[np.isnan(base_functions_test)] = 1/base_functions_test.shape[1]

    return base_functions_val, base_functions_test, y_val, y_test


if __name__ == "__main__":
    base_functions_val, base_functions_test, y_val, y_test = get_base_functions()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_samples,  num_classes, num_base_functions= base_functions_val.shape
    model = NeuralEnsembler(num_base_functions=num_base_functions,
                            num_classes=num_classes,
                            hidden_dim=32,
                            num_layers=3,
                            dropout_rate=0.2,
                            task_type="regression",
                            mode="stacking").to(device)

    trainer_args = TrainerArgs( batch_size=512, lr=0.001, epochs=1000, device=device)
    trainer = Trainer(model=model, trainer_args=trainer_args)
    trainer.fit(base_functions_val, y_val)
    y_pred_test = model.predict(base_functions_test)

    mae = mean_absolute_error(y_test, y_pred_test)
    print("mae", mae)

