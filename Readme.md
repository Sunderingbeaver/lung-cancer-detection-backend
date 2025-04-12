# Lung Cancer Detection

This is a monorepo containing the frontend and backend for the Lung Cancer Detection application.

## Project Structure

```
lung-cancer-detection/
├── Frontend/          # Vue.js frontend application
└── Backend/           # FastAPI backend application
```

## Frontend Development

The frontend is a Vue.js application located in the `Frontend` directory.

### Setup

1. Install dependencies:

```bash
npm install
```

2. Start development server:

```bash
npm run dev:frontend
```

3. Build for production:

```bash
npm run build:frontend
```

4. Preview production build:

```bash
npm run serve:frontend
```

## Backend Development

The backend is a FastAPI application located in the `Backend` directory.

### Setup

1. Install backend dependencies:

```bash
npm run install:backend
```

2. Start backend server:

```bash
npm run dev:backend
```

## Development Workflow

1. Start the backend server:

```bash
npm run dev:backend
```

2. In a new terminal, start the frontend development server:

```bash
npm run dev:frontend
```

The frontend will be available at `http://localhost:5173` and the backend at `http://localhost:8000`.

## Note

This repository uses Git submodules to manage the backend code. If you're not assigned to backend development:

-   You can still run the backend locally using the provided scripts
-   You won't receive backend updates unless you explicitly pull them
-   You can focus on frontend development without worrying about backend changes

Steps to run:

cd Backend
activate
venv\Scripts\python -m uvicorn api.backend:app --host 0.0.0.0 --port 8000 --reload

cd ..
cd Frontend/lung-cancer-frontend
npm run dev

#Currently only using v0.0.2b model
#Improve frontend
#Add the other 2 models
