import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import logging
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from whatsapp.handler import handle_incoming, load_history
from config.settings import GREEN_API_INSTANCE_ID, GREEN_API_TOKEN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title='Car Export Agent - WhatsApp Gateway', version='1.0.0')


@app.get('/health')
async def health() -> dict[str, Any]:
    green_api_configured = bool(GREEN_API_INSTANCE_ID and GREEN_API_TOKEN)
    return {
        'status': 'ok',
        'green_api_configured': green_api_configured,
    }


@app.post('/webhook')
async def webhook(request: Request) -> JSONResponse:
    try:
        payload: dict[str, Any] = await request.json()
    except Exception as exc:
        logger.warning('Invalid JSON received on /webhook: %s', exc)
        raise HTTPException(status_code=400, detail='Invalid JSON body') from exc

    webhook_type = payload.get('typeWebhook')
    logger.info('Received webhook type: %s', webhook_type)

    try:
        result = handle_incoming(payload)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception('Error processing webhook')
        raise HTTPException(status_code=500, detail='Failed to process webhook') from exc

    return JSONResponse(content=result)


@app.post('/simulate')
async def simulate(request: Request) -> JSONResponse:
    try:
        body: dict[str, Any] = await request.json()
    except Exception as exc:
        logger.warning('Invalid JSON received on /simulate: %s', exc)
        raise HTTPException(status_code=400, detail='Invalid JSON body') from exc

    phone = str(body.get('phone', '')).strip()
    name = str(body.get('name', '')).strip()
    message = str(body.get('message', '')).strip()

    if not message:
        raise HTTPException(status_code=422, detail='message must not be empty')

    chat_id = f'{phone}@c.us'

    payload: dict[str, Any] = {
        'typeWebhook': 'incomingMessageReceived',
        'senderData': {
            'chatId': chat_id,
            'sender': chat_id,
            'senderName': name,
        },
        'messageData': {
            'typeMessage': 'textMessage',
            'textMessageData': {
                'textMessage': message,
            },
        },
    }

    logger.info('Simulating incoming message for phone=%s name=%s', phone, name)

    try:
        result = handle_incoming(payload)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception('Error processing simulated webhook')
        raise HTTPException(status_code=500, detail='Failed to process simulated webhook') from exc

    return JSONResponse(content=result)


@app.get('/messages')
async def messages() -> JSONResponse:
    try:
        history = load_history()
    except Exception as exc:
        logger.exception('Error loading message history')
        raise HTTPException(status_code=500, detail='Failed to load message history') from exc

    return JSONResponse(content={'messages': history})


if __name__ == '__main__':
    uvicorn.run('server:app', host='0.0.0.0', port=8000, reload=False)