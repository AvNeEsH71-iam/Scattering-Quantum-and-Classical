# Scattering Explorer

Scattering Explorer is a scientific software package with two deliverables:

1. A Windows desktop application (`app/`) for classical and quantum scattering simulations.
2. A static website (`scattering-explorer-website/`) for showcasing and downloading the executable.

## Repository Layout

```
.
├── app/
│   ├── main.py
│   ├── simulations/
│   ├── ui/
│   ├── utils/
│   ├── assets/
│   ├── requirements.txt
│   └── README.md
├── scattering-explorer-website/
│   ├── index.html
│   ├── style.css
│   ├── script.js
│   ├── assets/
│   └── README.md
└── build.bat
```

## Quick Start

### Build desktop executable

```bat
build.bat
```

Output:

- `dist/ScatteringExplorer.exe`

### Website project

The download page is in:

- `scattering-explorer-website/`

Copy the latest executable into:

- `scattering-explorer-website/assets/ScatteringExplorer.exe`

## Documentation

- App build/run details: `app/README.md`
- Website deploy/update details: `scattering-explorer-website/README.md`
