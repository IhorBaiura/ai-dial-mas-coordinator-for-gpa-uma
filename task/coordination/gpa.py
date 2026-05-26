from copy import deepcopy
from typing import AsyncIterable, Optional, Any

from aidial_client import AsyncDial
from aidial_client.types.chat import ChatCompletionChunk
from aidial_sdk.chat_completion import Role, Choice, Request, Message, CustomContent, Stage, Attachment
from pydantic import StrictStr

from task.stage_util import StageProcessor

_IS_GPA = "is_gpa"
_GPA_MESSAGES = "gpa_messages"


class GPAGateway:

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    async def response(
            self,
            choice: Choice,
            stage: Stage,
            request: Request,
            additional_instructions: Optional[str]
    ) -> Message:
        client = AsyncDial(
            base_url=self.endpoint, 
            api_key=request.api_key,
            api_version='2025-01-01-preview'
        )

        gpa_messages = self.__prepare_gpa_messages(request=request, additional_instructions=additional_instructions)

        response: AsyncIterable[ChatCompletionChunk] = await client.chat.completions.create(
            deployment_name="general-purpose-agent",
            messages=gpa_messages,
            stream=True,
            extra_headers={
                'x-conversation-id': request.headers.get('X-Conversation-ID', 'unknown')
            }
        )   

        content = ""
        result_custom_content = CustomContent(attachments=[])
        stages_map: dict[int, Stage] = {}

        async for chunk in response:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                print(f"Received delta from GPA: {delta}")

                if delta and delta.content:
                    content += delta.content
                    stage.append_content(delta.content)

                if custom_content := delta.custom_content:
                    print(f"Received custom content from GPA: {custom_content}")

                    if custom_content.attachments:
                        result_custom_content.attachments.extend(custom_content.attachments)

                    if custom_content.state:
                        result_custom_content.state = custom_content.state

                    custom_content_dict = custom_content.dict(exclude_none=True)
                    if 'stages' in custom_content_dict:
                        for stage in custom_content_dict['stages']:
                            index = stage['index']
                            if opened_stage := stages_map.get(index):
                                if stage_content := stage.get("content"):
                                    opened_stage.append_content(stage_content)
                                elif stage_attachments := stage.get("attachments"):
                                    for stage_attachment in stage_attachments:
                                        opened_stage.add_attachment(Attachment(**stage_attachment))
                                elif stage.get("status") and stage.get("status") == 'completed':
                                    StageProcessor.close_stage_safely(stages_map[index])
                            else:
                                stages_map[index] = StageProcessor.open_stage(choice, stage.get("name"))

        for stage in stages_map.values():
            StageProcessor.close_stage_safely(stage)

        for attachment in result_custom_content.attachments:
            choice.add_attachment(Attachment(**attachment.dict(exclude_none=True)))

        choice.set_state({ _IS_GPA: True, _GPA_MESSAGES: result_custom_content.state })
        return Message(role=Role.ASSISTANT, content=content)

    def __prepare_gpa_messages(self, request: Request, additional_instructions: Optional[str]) -> list[dict[str, Any]]:
        res_messages: list[dict[str, Any]] = []

        for i in range(len(request.messages)):
            message = request.messages[i]
            if message.role == Role.ASSISTANT:
                if message.custom_content and message.custom_content.state:
                    message_state = message.custom_content.state
                    if message_state.get(_IS_GPA):
                        res_messages.append(request.messages[i-1].dict(exclude_none=True))
                        copy = deepcopy(message)
                        copy.custom_content.state = message_state.get(_GPA_MESSAGES)
                        res_messages.append(copy.dict(exclude_none=True))

        last_user_message = request.messages[-1]

        if additional_instructions:
            custom_content = last_user_message.custom_content
            res_messages.append(
                {
                    "role": Role.USER,
                    "content": f"{last_user_message.content}\n\n{additional_instructions}",
                    "custom_content": custom_content.dict(exclude_none=True) if custom_content else None,
                }
            )
        else:
            res_messages.append(last_user_message.dict(exclude_none=True))

        return res_messages
