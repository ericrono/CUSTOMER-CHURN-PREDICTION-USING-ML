# Quick Start Guide

## Get Up and Running in 5 Minutes

### Step 1: Setup Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Configure Database (Optional)
If you need database access:
```bash
# Copy template
copy .env.example .env

# Edit .env with your credentials
notepad .env
```

### Step 4: Launch Jupyter
```bash
cd codes
jupyter notebook
```

### Step 5: Run the Notebook
Open `lp2 - customer churn classification.ipynb` and run all cells!

---

## Troubleshooting

### Import Error?
```bash
pip install scikit-learn==1.5.1 imbalanced-learn==0.12.3
```

### Can't activate venv on Windows?
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Database connection failing?
- Check SQL Server is running
- Verify credentials in .env file
- Ensure ODBC Driver is installed

---

## What's Included

- 📊 **Customer Churn Analysis** - Complete EDA and insights
- 🤖 **ML Models** - Multiple classifiers with hyperparameter tuning
- 📈 **Visualizations** - Interactive plots and charts
- 📉 **Model Evaluation** - Confusion matrices and metrics
- 💡 **Business Recommendations** - Actionable insights

---

## Need More Help?
- See `SETUP.md` for detailed instructions
- Check `FIXES_APPLIED.md` for known issues and solutions
- Review `README.md` for project overview

Happy analyzing! 🚀
