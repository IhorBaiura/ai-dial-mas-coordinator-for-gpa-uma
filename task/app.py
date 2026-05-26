import os

import uvicorn
from aidial_sdk import DIALApp
from aidial_sdk.chat_completion import ChatCompletion, Request, Response

from task.agent import MASCoordinator
from task.logging_config import setup_logging, get_logger

DIAL_ENDPOINT = os.getenv('DIAL_ENDPOINT', "http://localhost:8080")
DEPLOYMENT_NAME = os.getenv('DEPLOYMENT_NAME', 'gpt-4o')
UMS_AGENT_ENDPOINT = os.getenv('UMS_AGENT_ENDPOINT', "http://localhost:8042")
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

setup_logging(log_level=LOG_LEVEL)
logger = get_logger(__name__)


class MASCoordinatorApplication(ChatCompletion):
    '''
    MAS Coordinator Application that handles chat completion requests and coordinates multiple agents.
    '''

    async def chat_completion(self, request: Request, response: Response) -> None:
        '''Handle chat completion request and coordinate multiple agents.'''

        conversation_id = request.headers.get("X-Conversation-ID", "unknown")
        logger.info(f"Received chat completion request for conversation_id: {conversation_id}")

        try:
            with response.create_single_choice() as single_choice:
                logger.info(f"Created single choice for conversation_id: {conversation_id}")

                mas_coordinator = MASCoordinator(
                    endpoint=DIAL_ENDPOINT,
                    deployment_name=DEPLOYMENT_NAME,
                    ums_agent_endpoint=UMS_AGENT_ENDPOINT
                )
                await mas_coordinator.handle_request(choice=single_choice, request=request)

                logger.info(f"Handled chat completion request for conversation_id: {conversation_id} with MAS Coordinator")

        except Exception as e:
            logger.error(f"Error handling chat completion request: {e}")
            raise e


logger.info(f"Creating MAS Coordinator Application with DIAL endpoint: {DIAL_ENDPOINT}, \
            deployment name: {DEPLOYMENT_NAME}, UMS agent endpoint: {UMS_AGENT_ENDPOINT}")
app = DIALApp()
coordinator = MASCoordinatorApplication()
app.add_chat_completion(deployment_name="mas-coordinator", impl=coordinator)

if __name__ == "__main__":
    logger.info("Running MAS Coordinator Application on 0.0.0.0:8055...")
    uvicorn.run(app, port=8055, host="0.0.0.0", log_level=LOG_LEVEL.lower())
