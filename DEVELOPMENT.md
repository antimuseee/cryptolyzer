# 🚀 Development Guide - Real-Time Development

## Quick Start

### Option 1: Windows (Recommended)
Double-click `dev.bat` or run:
```bash
dev.bat
```

### Option 2: Command Line
```bash
python dev_server.py
```

### Option 3: Standard Flask (Basic)
```bash
python app.py
```

## 🔄 Real-Time Development Features

### What Auto-Reloads:
- ✅ **Python files** (`.py`) - Flask debug mode handles this
- ✅ **HTML templates** - Flask debug mode handles this
- ⚠️ **Static files** (CSS/JS) - Manual browser refresh needed

### What You'll See:
1. **Server Console**: Shows file changes and reload notifications
2. **Browser**: Automatically refreshes when Python/HTML files change
3. **File Watcher**: Monitors all project files and shows change notifications

## 🛠️ Development Workflow

1. **Start the server** using one of the methods above
2. **Open browser** to `http://localhost:5000`
3. **Make changes** to your files
4. **Watch the magic** - changes appear automatically!

### File Change Notifications:
```
🔄 File changed: app.py
💡 Flask debug mode will auto-reload the server
🌐 Refresh your browser to see changes
```

## 📁 File Structure for Development

```
cryptolyzer/
├── app.py              # Main Flask app (auto-reloads)
├── cryptolyzer.py      # Analysis logic (auto-reloads)
├── dev_server.py       # Enhanced dev server
├── dev.bat            # Windows quick start
├── templates/
│   └── index.html     # HTML templates (auto-reloads)
├── static/
│   ├── js/
│   │   ├── charts.js  # JavaScript (manual refresh)
│   │   └── matrix-bg.js
└── requirements.txt    # Dependencies
```

## 🔧 Troubleshooting

### If auto-reload isn't working:
1. Make sure you're using `dev_server.py` or `dev.bat`
2. Check that `debug=True` is set in `app.py`
3. Ensure you're editing files in the project directory

### If browser doesn't refresh:
1. Manual refresh: `Ctrl+F5` (hard refresh)
2. Check browser console for errors
3. Clear browser cache

### If server won't start:
1. Install dependencies: `pip install -r requirements.txt`
2. Check if port 5000 is available
3. Try a different port in `app.py`

## 🎯 Best Practices

1. **Use the dev server** (`dev_server.py`) for the best experience
2. **Keep browser open** while developing
3. **Check console output** for error messages
4. **Use hard refresh** (`Ctrl+F5`) for static file changes

## 🚀 Production vs Development

- **Development**: Use `dev_server.py` with auto-reload
- **Production**: Use `gunicorn` or standard Flask deployment

---

**Happy coding! 🎉** 