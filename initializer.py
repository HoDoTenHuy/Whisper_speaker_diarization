class IncludeAPIRouter(object):
    def __new__(cls):
        from routers.health_check import router as router_health_check
        from routers.speech_to_text import router as router_speech2text
        from fastapi.routing import APIRouter

        router = APIRouter()
        router.include_router(router_health_check, prefix='/api/v1', tags=['Health Check'])
        router.include_router(router_speech2text, prefix='/api/v1', tags=['Speech to Text'])
        return router
