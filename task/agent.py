import json
from copy import deepcopy
from typing import Any

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Role, Choice, Request, Message, Stage
from pydantic import StrictStr

from task.coordination.gpa import GPAGateway
from task.coordination.ums_agent import UMSAgentGateway
from task.logging_config import get_logger
from task.models import CoordinationRequest, AgentName
from task.prompts import COORDINATION_REQUEST_SYSTEM_PROMPT, FINAL_RESPONSE_SYSTEM_PROMPT
from task.stage_util import StageProcessor

logger = get_logger(__name__)


class MASCoordinator:

    def __init__(self, endpoint: str, deployment_name: str, ums_agent_endpoint: str):
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.ums_agent_endpoint = ums_agent_endpoint

    async def handle_request(self, choice: Choice, request: Request) -> Message:
        client = AsyncDial(
            base_url=self.endpoint, 
            api_key=request.api_key,
            api_version='2025-01-01-preview'
        )

        stage = StageProcessor.open_stage(choice=choice, name="Coordination Request")
        coordination_request = await self.__prepare_coordination_request(client=client, request=request)

        logger.info(f"Prepared coordination request for conversation_id: {request.headers.get('X-Conversation-ID', 'unknown')}, \
                    coordination_request: {coordination_request.model_dump_json()}")
        
        stage.append_content(f"```json\n\n{coordination_request.model_dump_json(indent=2)}\n\n```\n")
        StageProcessor.close_stage_safely(stage=stage)

        stage = StageProcessor.open_stage(choice=choice, name=f"Call: `{coordination_request.agent_name}` agent")
        agent_message = await self.__handle_coordination_request(
            coordination_request=coordination_request,
            choice=choice,
            stage=stage,
            request=request
        )

        logger.info(f"Received response from agent for conversation_id: {request.headers.get('X-Conversation-ID', 'unknown')}, \
                    agent_message: {agent_message.model_dump_json()}")
        StageProcessor.close_stage_safely(stage=stage)

        final_response = await self.__final_response(
            client=client,
            choice=choice,
            request=request,
            agent_message=agent_message
        )

        logger.info(f"Prepared final response for conversation_id: {request.headers.get('X-Conversation-ID', 'unknown')}, \
                    final_response: {final_response.model_dump_json()}")

        return final_response

    async def __prepare_coordination_request(self, client: AsyncDial, request: Request) -> CoordinationRequest:
        response = await client.chat.completions.create(
            deployment_name=self.deployment_name,
            messages=self.__prepare_messages(request=request, system_prompt=COORDINATION_REQUEST_SYSTEM_PROMPT),
            extra_body={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": CoordinationRequest.model_json_schema()
                    }
                }
            }
        )

        content = response.choices[0].message.content
        if isinstance(content, StrictStr):
            content_dict = json.loads(content)
            return CoordinationRequest.model_validate(content_dict)
        else:
            raise ValueError(f"Expected content to be a string, got {type(content)}")

    def __prepare_messages(self, request: Request, system_prompt: str) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": Role.SYSTEM, "content": system_prompt}
        ]

        for message in request.messages:
            if message.role == Role.USER and message.custom_content is not None:
                messages.append({"role": Role.USER, "content": deepcopy(message).content})
            else:
                messages.append(deepcopy(message).dict(exclude_none=True))
                
        return messages

    async def __handle_coordination_request(
            self,
            coordination_request: CoordinationRequest,
            choice: Choice,
            stage: Stage,
            request: Request
    ) -> Message:
        match coordination_request.agent_name:
            case AgentName.GPA:
                gpa_gateway = GPAGateway(endpoint=self.endpoint)
                return await gpa_gateway.response(
                    choice=choice,
                    request=request,
                    stage=stage,
                    additional_instructions=coordination_request.additional_instructions
                )
            case AgentName.UMS:
                ums_agent_gateway = UMSAgentGateway(ums_agent_endpoint=self.ums_agent_endpoint)
                return await ums_agent_gateway.response(
                    choice=choice,
                    request=request,
                    stage=stage,
                    additional_instructions=coordination_request.additional_instructions
                )
            case _:
                logger.error(f"Unknown agent name: {coordination_request.agent_name} for conversation_id: {request.headers.get('X-Conversation-ID', 'unknown')}")
                raise ValueError(f"Unknown agent name: {coordination_request.agent_name}")

    async def __final_response(
            self, client: AsyncDial,
            choice: Choice,
            request: Request,
            agent_message: Message
    ) -> Message:
        messages: list[dict[str, Any]] = self.__prepare_messages(request=request, system_prompt=FINAL_RESPONSE_SYSTEM_PROMPT)

        augmented_user_request = f"## CONTEXT:\n {agent_message.content}\n ---\n ## USER_REQUEST: \n {messages[-1]["content"]}"
        messages[-1]["content"] = augmented_user_request

        chunks = await client.chat.completions.create(
            deployment_name=self.deployment_name,
            messages=messages,
            stream=True
        )

        content = ''
        async for chunk in chunks:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    choice.append_content(delta.content)
                    content += delta.content

        return Message(
            role=Role.ASSISTANT,
            custom_content=agent_message.custom_content,
            content=content
        )
