# CoachAI+ Frontend
Web page for CoachAI+

## Requirement
- `Node.js` > 18, 20 is recommended.
- Node package manager, e.g. `npm`. `pnpm` is recommended.
- API server should host on `http://localhost:8000`, or you can configre in `vite.condig.ts`.
- Create `.env` file and add following settings
  ```
  VITE_API_BASE=http://127.0.0.1:8000
  ```

## Installation
```bash
pnpm run install
```

## Run dev server
```bash
pnpm run dev
```
Dev server will be available at http://localhost:5173

## Run preview server
```bash
pnpm run preview
```

## Build website
```bash
pnpm run build
```