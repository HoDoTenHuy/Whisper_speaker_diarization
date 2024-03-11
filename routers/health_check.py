import logging

from fastapi.routing import APIRouter


router = APIRouter()
logger = logging.getLogger()


@router.get('/ping')
async def health_check() -> dict:
    logger.info('event={} message="{}"'.format('health-check-success', 'Successful health check.'))
    return {'OKE': True}
