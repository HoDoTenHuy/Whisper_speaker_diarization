import uvicorn
import logging

from fastapi import FastAPI
from starlette.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from initializer import IncludeAPIRouter
from utils.config_yaml import ConfigManager
from contextlib import asynccontextmanager

config_manager = ConfigManager()
config = config_manager.get_config()
api_config = config.get('api')

logger = logging.getLogger(__name__)


def get_application():
    _app = FastAPI(title=api_config.get('name'),
                   description=api_config.get('description'),
                   version=api_config.get('version'),
                   debug=api_config.get('debug_mode'))
    _app.include_router(IncludeAPIRouter())
    _app.add_middleware(
        CORSMiddleware,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return _app


app = get_application()


@app.get("/")
async def docs_redirect():
    return RedirectResponse(url='/docs')


@app.on_event("shutdown")
async def app_shutdown():
    # on app shutdown do something probably close some connections or trigger some event
    logger.info('event={} message="{}"'.format('app-shutdown', 'All connections are closed.'))


if __name__ == '__main__':
    uvicorn.run("app:app",
                host=api_config.get('host'),
                port=api_config.get('port'),
                reload=True)
