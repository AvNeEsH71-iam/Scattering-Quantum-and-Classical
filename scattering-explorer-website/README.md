# Scattering Explorer Website

Static landing page and download portal for `ScatteringExplorer.exe`.

## Structure

```
scattering-explorer-website/
├── index.html
├── style.css
├── script.js
└── assets/
    ├── ScatteringExplorer.exe
    └── demo.mp4
```

## Local Preview

From repository root:

```bat
cd scattering-explorer-website
python -m http.server 5500
```

Then open:

- `http://localhost:5500`

## Update EXE Download

1. Build latest app executable:

```bat
cd ..
build.bat
```

2. Copy built executable into website assets:

```bat
copy /y dist\ScatteringExplorer.exe scattering-explorer-website\assets\ScatteringExplorer.exe
```

3. Commit and push changes.

The download link is already configured as:

```html
<a href="assets/ScatteringExplorer.exe" download>Download for Windows</a>
```

## Deploy to GitHub Pages

1. Create a new GitHub repository named `scattering-explorer-website`.
2. Copy the contents of this folder (`index.html`, `style.css`, `script.js`, `assets/`) into that repo root.
3. Push to `main` branch.
4. Open repository settings:
   - `Settings` -> `Pages`
   - `Build and deployment` -> `Source: Deploy from a branch`
   - `Branch: main`
   - `Folder: / (root)`
5. Save settings and wait for deployment.

Final URL format:

- `https://<username>.github.io/scattering-explorer-website/`

## GitHub Section Link

Update the placeholder source URL in `index.html`:

- Replace `https://github.com/<username>/scattering-explorer` with your actual repository URL.
