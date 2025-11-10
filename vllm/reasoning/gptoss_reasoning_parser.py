# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from typing import Any
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ResponsesRequest, ChatCompletionToolsParam
from collections.abc import Sequence

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.harmony_utils import parse_chat_output
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, DeltaMessage
from vllm.entrypoints.tool_server import ToolServer
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser

logger = init_logger(__name__)

no_func_reaonsing_tag = {
    "type": "structural_tag",
    "format": {
        "type": "triggered_tags",
        "tags": [
            {
                "begin": "<|channel|>analysis<|message|>",
                "content": {"type": "any_text"},
                "end": "<|end|>",
            }
        ],
        "triggers": ["<|channel|>analysis"],
        "stop_after_first": False,
    },
}


def from_builtin_tool_to_tag(tool: str) -> list[dict]:
    tag = [
        {
            "begin": f"<|channel|>commentary to={tool}",
            "content": {"type": "any_text"},
            "end": "<|end|>",
        },
        {
            "begin": f"<|channel|>analysis to={tool}",
            "content": {"type": "any_text"},
            "end": "<|end|>",
        },
    ]
    return tag


def tag_with_builtin_funcs(no_func_reaonsing_tag, builtin_tool_list: list[str]) -> dict:
    import copy

    new_tag = copy.deepcopy(no_func_reaonsing_tag)
    new_tag["format"]["triggers"].append("<|channel|>commentary to=")

    for tool in builtin_tool_list:
        new_tag["format"]["tags"].extend(from_builtin_tool_to_tag(tool))
    return new_tag


def from_custom_tool_to_tag(tool: ChatCompletionToolsParam) -> dict | None:
    """Создает правило грамматики для одного кастомного инструмента."""
    if tool.type != "function":
        return None

    return {
        "begin": f"<|channel|>commentary to=functions.{tool.function.name}",
        "schema": tool.function.parameters,
        "end": "<|call|>"
    }


def tag_with_custom_tools(grammar: dict, custom_tools: list[ChatCompletionToolsParam]) -> dict:
    """Расширяет грамматику правилами для кастомных инструментов."""
    import copy
    new_grammar = copy.deepcopy(grammar)

    if not custom_tools:
        return new_grammar

    if "<|channel|>commentary to=" not in new_grammar["format"]["triggers"]:
        new_grammar["format"]["triggers"].append("<|channel|>commentary to=")

    for tool in custom_tools:
        tool_rule = from_custom_tool_to_tag(tool)
        if tool_rule:
            new_grammar["format"]["tags"].append(tool_rule)

    return new_grammar


def tag_with_final_response_schema(grammar: dict, schema: dict[str, Any]) -> dict:
    """Расширяет грамматику правилом для структурированного финального ответа."""
    import copy
    new_grammar = copy.deepcopy(grammar)

    final_tag_rule = {
        "begin": "<|channel|>final<|message|>",
        "schema": schema,
        "end": "<|return|>"
    }
    new_grammar["format"]["tags"].append(final_tag_rule)

    if "<|channel|>final" not in new_grammar["format"]["triggers"]:
        new_grammar["format"]["triggers"].append("<|channel|>final")

    return new_grammar


class GptOssReasoningParser(ReasoningParser):
    """
    Reasoning parser for GptOss model.

    The GptOss model uses harmony to extract reasoning content and this parser
    is only used for detecting the end of the reasoning content.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        # The model can output some special tokens between "final" and "<|message|>"
        # So we need to look for both sequences to determine the end of reasoning.
        self.reasoning_end_token_ids_prefix = self.model_tokenizer.encode(
            "<|channel|>final"
        )
        self.reasoning_end_token_ids_suffix = self.model_tokenizer.encode("<|message|>")
        self.reasoning_max_num_between_tokens = 20

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        end_token_ids_prefix = self.reasoning_end_token_ids_prefix
        end_token_ids_suffix = self.reasoning_end_token_ids_suffix
        assert len(end_token_ids_prefix) > 0, "reasoning_end_token_ids_prefix is empty"
        assert len(end_token_ids_suffix) > 0, "reasoning_end_token_ids_suffix is empty"
        # Check if the end sequence is present in the input_ids.
        # We search from the end of input_ids to find the last match.
        for i in range(len(input_ids) - len(end_token_ids_prefix), -1, -1):
            if input_ids[i : i + len(end_token_ids_prefix)] == end_token_ids_prefix:
                # We have found the prefix, now we look for the suffix after the prefix.
                suffix_start = i + len(end_token_ids_prefix)
                for j in range(
                    suffix_start, len(input_ids) - len(end_token_ids_suffix) + 1
                ):
                    if j - suffix_start >= self.reasoning_max_num_between_tokens:
                        break
                    if (
                        input_ids[j : j + len(end_token_ids_suffix)]
                        == end_token_ids_suffix
                    ):
                        return True
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        _, content, _ = parse_chat_output(input_ids)
        if content is None:
            return []
        return self.model_tokenizer.encode(content)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        prev_reasoning, prev_content, _ = parse_chat_output(list(previous_token_ids))
        cur_reasoning, cur_content, _ = parse_chat_output(list(current_token_ids))
        reasoning_delta = None
        content_delta = None
        if cur_reasoning is not None:
            prev_r = prev_reasoning or ""
            if cur_reasoning.startswith(prev_r):
                reasoning_delta = cur_reasoning[len(prev_r) :] or None
            else:
                reasoning_delta = cur_reasoning
        if cur_content is not None:
            prev_c = prev_content or ""
            if cur_content.startswith(prev_c):
                content_delta = cur_content[len(prev_c) :] or None
            else:
                content_delta = cur_content
        if reasoning_delta is None and content_delta is None:
            return None
        return DeltaMessage(reasoning=reasoning_delta, content=content_delta)

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> tuple[str | None, str | None]:
        raise NotImplementedError(
            "gpt-oss has a special branch for parsing reasoning in non-streaming mode. This method shouldn't be used."  # noqa: E501
        )

    # This function prepares the structural tag to format reasoning output
    def prepare_structured_tag(
            self,
            original_tag: str | None,
            tool_server: ToolServer | None,
            request: ChatCompletionRequest | ResponsesRequest | None = None,
    ) -> str:
        """
                Создает грамматику `structural_tag` для управления выводом модели gpt-oss.
                Эта грамматика динамически строится на основе:
                1. Наличия встроенных инструментов (browser, python).
                2. Кастомных инструментов, переданных в запросе.
                3. JSON-схемы для финального ответа, переданной в запросе.
                """
        if original_tag is not None:
            return original_tag

        custom_tools: list[ChatCompletionToolsParam] | None = None
        final_response_schema: dict[str, Any] | None = None

        if request:
            # 1. Извлекаем кастомные инструменты (поле 'tools' есть в обоих типах)
            if request.tools:
                # Фильтруем, чтобы оставить только 'function' инструменты,
                # так как gpt-oss понимает только их в этом контексте.
                # Также проверяем наличие атрибутов для безопасности.
                custom_tools = [
                    tool for tool in request.tools
                    if (hasattr(tool, 'type') and tool.type == "function" and
                        hasattr(tool, 'function') and hasattr(tool.function, 'name'))
                ]

            # 2. Извлекаем схему для финального ответа (только для ResponsesRequest)
            # Проверяем, что это объект ResponsesRequest и нужные поля существуют.
            if isinstance(request, ResponsesRequest) and hasattr(request, 'text') and request.text:
                if (
                        request.text.format is not None and
                        request.text.format.type == "json_schema" and
                        hasattr(request.text.format, 'schema_') and
                        request.text.format.schema_ is not None
                ):
                    final_response_schema = request.text.format.schema_

        import copy
        # Начинаем с базовой грамматики, которая разрешает только 'analysis'
        current_grammar = copy.deepcopy(no_func_reaonsing_tag)

        # 3. Добавляем встроенные инструменты, если они есть и активны
        if tool_server is not None:
            builtin_tool_list: list[str] = [
                tool for tool in ["browser", "python", "container"] if tool_server.has_tool(tool)
            ]
            if builtin_tool_list:
                logger.info("Adding builtin tools to grammar: %s", builtin_tool_list)
                current_grammar = tag_with_builtin_funcs(current_grammar, builtin_tool_list)

        # 4. Добавляем кастомные инструменты, если они были извлечены из запроса
        if custom_tools:
            logger.info("Adding %d custom tools to grammar.", len(custom_tools))
            current_grammar = tag_with_custom_tools(current_grammar, custom_tools)

        # 5. Добавляем схему для финального ответа, если она была извлечена
        if final_response_schema:
            logger.info("Adding final response schema to grammar.")
            current_grammar = tag_with_final_response_schema(current_grammar, final_response_schema)

        return json.dumps(current_grammar)
