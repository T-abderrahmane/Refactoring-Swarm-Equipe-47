# Quick Start Guide - Refactoring Swarm

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Dependencies

```bash
# Activate virtual environment (if not already)
source venv/bin/activate  # Linux/Mac
# OR
.\venv\Scripts\activate  # Windows

# Install packages
pip install -r requirements.txt
```

### Step 2: Set Up API Key

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your Google Gemini API key
# Get a free key at: https://makersuite.google.com/app/apikey
nano .env  # or use your favorite editor
```

Your `.env` should look like:
```
GOOGLE_API_KEY=AIzaSy...your_actual_key_here...
```

### Step 3: Test the System

Run on the included test code:

```bash
python main_new.py
```

This will:
1. ✅ Analyze `sandbox/test_code.py`
2. ✅ Generate a refactoring plan
3. ✅ Apply fixes
4. ✅ Run tests
5. ✅ Show results

### Step 4: Use on Your Code

```bash
# Refactor your own code
python main_new.py --target_dir ./path/to/your/code

# Your code will be copied to ./sandbox first (safe!)
# Original files are never modified
```

## 📊 Understanding the Output

The system shows real-time progress:

```
=== ANALYZER AGENT ===
🔍 Scanning directory...
✅ Found 5 Python files
🔍 Running static analysis...
🤖 Generating refactoring plan...

=== FIXER AGENT ===
🔧 Processing file 1/5...
✅ Fixed and saved

=== JUDGE AGENT ===
🧪 Running tests...
✅ Tests PASSED
```

## 📁 Output Files

After running, check:

- **Fixed code**: `./sandbox/` - Your refactored code
- **Logs**: `./logs/experiment_data.json` - All LLM interactions
- **Console**: Real-time progress and summary

## ⚙️ Common Options

```bash
# Limit iterations (default: 10)
python main_new.py --max_iterations 5

# Use specific directory
python main_new.py --target_dir ./my_project

# Run directly on sandbox (skip copy)
python main_new.py --no_copy
```

## 🔧 Troubleshooting

### "GOOGLE_API_KEY not found"
- Make sure you created `.env` file
- Check that the key starts with `AIzaSy`
- Ensure `.env` is in the project root

### "No Python files found"
- Check that your target directory contains `.py` files
- Make sure you're pointing to the right directory

### "Pylint not found" or "Pytest not found"
- Run: `pip install pylint pytest`
- Or reinstall: `pip install -r requirements.txt`

## 📚 Next Steps

1. Read the full [README_NEW.md](README_NEW.md)
2. Check [docs/](docs/) for project requirements
3. Customize agents for your needs
4. Run on your real projects!

## 💡 Tips

- Start with small projects (1-5 files)
- Review the refactored code before using
- Check `logs/experiment_data.json` to see what the AI did
- Use `--max_iterations 3` for faster testing

---

Need help? Check the documentation or ask your instructor!
