# Campus Voice

Campus Voice is a Flask web application for collecting and managing student feedback. Students can submit feedback, review its status, and exchange messages with administrators. A TensorFlow text classifier automatically assigns each submission to a topic.

## Features

- Student registration and login, with role-based access for administrators.
- Feedback submission with automatic topic classification into lecturer, curriculum, facilities, or other.
- Confidence thresholding that assigns low-confidence predictions to the `Other` category.
- Student and administrator feedback threads, status tracking, and administrator search/category filters.
- An administrator dashboard with monthly feedback totals by category.
- SQLite persistence; the database and tables are initialized when the application starts.

## Requirements

- Python version supported by the installed TensorFlow release.
- The packages listed in `requirements.txt`.

Install the dependencies in a virtual environment:

```powershell
python -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead.

## Run the application

```powershell
python app.py
```

Open `http://127.0.0.1:5001`. The application creates `feedback.db` in the project directory and seeds an administrator account if one does not already exist:

- Email: `admin@campus.local`
- Password: `admin123`

These credentials are for local development only. Do not expose the application publicly with the default secret key or seeded password.

Set `SECRET_KEY` to a random secret before deployment. `HOST` and `PORT` can optionally override the bind address and port.

## Model and data

The classifier implementation is in `test.py`. It tokenizes Vietnamese text with Underthesea and classifies it into four topics. The trained model and vocabulary are stored in `models/topic_classifier.keras` and `models/vocabulary.json`; the app loads these artifacts when making predictions.

Training and validation data are read from `data/train.csv` and `data/validation.csv`. Both files are expected to contain:

- `sentence`: feedback text.
- `topic`: integer class ID (`0` lecturer, `1` curriculum, `2` facilities, `3` other).

To train and save a new model using the training and validation CSV files:

```powershell
python test.py --train
```

Training also writes `result_train.txt` in the project directory with accuracy, macro precision, macro recall, macro F1-score, and per-topic metrics for the training, validation, and final test datasets. Use the `Final Test` section as the final evaluation; do not use test results to tune the model.

The optional `--epochs` argument sets the training epoch limit, for example `python test.py --train --epochs 20`. Predictions are made by the web application as students submit feedback.

## Main routes

| Route | Purpose |
| --- | --- |
| `/register` | Create a student account |
| `/login` | Sign in |
| `/logout` | Sign out |
| `/` | Student feedback dashboard and submission form |
| `/admin` | Administrator dashboard, feedback search, and category filters |
| `/admin/feedback/<id>/respond` | Post an administrator reply to a feedback thread |
| `/api/my-feedback-stats` | Return the signed-in student's feedback counts as JSON |
