# CoachAI+ Backend
Simple WSGI server using `fastapi`

## Run dev server
Go to parent folder to start server
```sh
fastapi dev api/app.py
```

## Run prod server
Go to parent folder to start server
```sh
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

## API Document

After start the server, swagger page will auto start at `/docs`

http://localhost:8000/docs