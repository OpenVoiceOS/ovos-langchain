import os
import uuid
from threading import Event
from typing import Any

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms.base import LLM

_ff_account_data = None
_usesless_account_data = None
_quora_acc = None


class GPT4Free(LLM):

    @staticmethod
    def ask_gpt4free(prompt):
        bad = ["retry", "Unable to fetch the response, Please try again."]
        r = None

        funcs = [
            GPT4Free.ask_gpt4free_aiassist,
            GPT4Free.ask_gpt4free_you,
            GPT4Free.ask_gpt4free_deepai,
            #GPT4Free.ask_gpt4quora,
            #GPT4Free.ask_gpt4free_gptworldai,
            #GPT4Free.ask_gpt4free_usesless,
            #GPT4Free.ask_gpt4free_theb,
            #GPT4Free.ask_gpt4free_forefront
        ]
        for proxy in funcs:
            print(proxy)
            try:
                r = proxy(prompt)
            except Exception as e:
                print(e)
            if r and r not in bad:
                return r

        return
        try:
            r = NeonLLM.ask_neon(prompt)
        except:
            pass
        if r and r not in bad:
            return r

    @staticmethod
    def ask_gpt4quora(prompt, model='gpt-3.5-turbo'):
        global _quora_acc
        from gpt4free import quora
        if not _quora_acc:
            _quora_acc = quora.Account.create(logging=True, enable_bot_creation=True)
        # simple request with links and details
        response = quora.Completion.create(prompt=prompt,
                                           token=_quora_acc,
                                           model=model)
        return response

    @staticmethod
    def ask_gpt4freehpgptdai(prompt):
        from gpt4free import hpgptai
        # simple request with links and details
        response = hpgptai.Completion.create(prompt=prompt, proxy=None)
        return response

    @staticmethod
    def ask_gpt4free_gptworldai(prompt):
        from gpt4free import gptworldAi
        # simple request with links and details
        response = gptworldAi.Completion.create(prompt=prompt)
        return "".join(response)

    @staticmethod
    def ask_gpt4free_aiassist(prompt):
        from gpt4free import aiassist

        # simple request with links and details
        response = aiassist.Completion.create(prompt=prompt)
        return response["text"]

    @staticmethod
    def ask_gpt4free_you(prompt):
        from gpt4free import you

        # simple request with links and details
        response = you.Completion.create(prompt=prompt)
        t = response.text
        return t

    @staticmethod
    def ask_gpt4free_deepai(prompt):
        from gpt4free import deepai
        # simple request with links and details
        response = deepai.Completion.create(prompt=prompt)
        return "".join(response)

    @staticmethod
    def ask_gpt4free_theb(prompt):
        from gpt4free import theb
        # simple request with links and details
        response = theb.Completion.create(prompt=prompt)
        return "".join(response)

    @staticmethod
    def ask_gpt4free_usesless(prompt):
        global _usesless_account_data
        from gpt4free import usesless
        # simple request with links and details
        # create an account
        if not _usesless_account_data:
            _usesless_account_data = usesless.Account.create(logging=False)
            print(_usesless_account_data)
        # get a response
        res = usesless.Completion.create(
            token=_usesless_account_data,
            prompt=prompt
        )
        return res['text']

    @staticmethod
    def ask_gpt4free_forefront(prompt, model='gpt-4'):
        global _ff_account_data
        from gpt4free import forefront
        # simple request with links and details
        # create an account
        if not _ff_account_data:
            _ff_account_data = forefront.Account.create(logging=False)

        # get a response
        for response in forefront.StreamingCompletion.create(
                account_data=_ff_account_data,
                prompt=prompt,
                model=model
        ):
            return response.choices[0].text


class NeonLLM(LLM):

    def _call(
            self,
            prompt: str,
            stop=None,
            run_manager=None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return self.ask_neon(prompt)

    @property
    def _llm_type(self) -> str:
        return "custom"

    @property
    def _identifying_params(self):
        """Get the identifying parameters."""
        return {}

    @staticmethod
    def send_mq_request(vhost: str, request_data: dict, target_queue: str,
                        response_queue: str = None, timeout: int = 30,
                        expect_response: bool = True) -> dict:
        """
        Sends a request to the MQ server and returns the response.
        :param vhost: vhost to target
        :param request_data: data to post to target_queue
        :param target_queue: queue to post request to
        :param response_queue: optional queue to monitor for a response.
            Generally should be blank
        :param timeout: time in seconds to wait for a response before timing out
        :param expect_response: boolean indicating whether a response is expected
        :return: response to request
        """
        from neon_mq_connector.utils.network_utils import b64_to_dict
        from neon_mq_connector.utils.client_utils import NeonMQHandler
        from pika.channel import Channel
        from pika.exceptions import StreamLostError

        response_queue = response_queue or uuid.uuid4().hex

        response_event = Event()
        message_id = None
        response_data = dict()

        def on_error(thread, error):
            """
            Override default error handler to suppress certain logged errors.
            """
            if isinstance(error, StreamLostError):
                return
            print(f"{thread} raised {error}")

        def handle_mq_response(channel: Channel, method, _, body):
            """
                Method that handles Neon API output.
                In case received output message with the desired id, event stops
            """
            api_output = b64_to_dict(body)
            api_output_msg_id = api_output.get('message_id', None)
            if api_output_msg_id == message_id:
                channel.basic_ack(delivery_tag=method.delivery_tag)
                channel.close()
                response_data.update(api_output)
                response_event.set()
            else:
                channel.basic_nack(delivery_tag=method.delivery_tag)

        try:
            mq_config = {
                "server": "mq.2023.us",
                "port": 35672,
                "users": {
                    "mq_handler": {
                        "user": 'neon_api_utils',
                        "password": 'Klatchat2021'
                    }
                }
            }
            neon_api_mq_handler = NeonMQHandler(config=mq_config,
                                                service_name='mq_handler',
                                                vhost=vhost)
            if not neon_api_mq_handler.connection.is_open:
                raise ConnectionError("MQ Connection not established.")

            if expect_response:
                neon_api_mq_handler.register_consumer(
                    'neon_output_handler', neon_api_mq_handler.vhost,
                    response_queue, handle_mq_response, on_error, auto_ack=False)
                neon_api_mq_handler.run_consumers()
                request_data['routing_key'] = response_queue

            message_id = neon_api_mq_handler.emit_mq_message(
                connection=neon_api_mq_handler.connection, queue=target_queue,
                request_data=request_data, exchange='')

            if expect_response:
                response_event.wait(timeout)
                if not response_event.is_set():
                    print(f"Timeout waiting for response to: {message_id} on "
                          f"{response_queue}")
                neon_api_mq_handler.stop_consumers()
        except Exception as ex:
            print(f'Exception occurred while resolving Neon API: {ex}')
        return response_data

    @staticmethod
    def ask_neon(query):
        mq_resp = NeonLLM.send_mq_request("/llm",
                                          {"query": query, "history": []},
                                          "chat_gpt_input")
        return mq_resp.get("response") or ""


class RWKVLLM(LLM):
    model: Any = None

    @classmethod
    def init(cls, model, tokens_path, strategy="cpu fp8"):
        from langchain.llms import RWKV
        cls.model = RWKV(model=model, strategy=strategy, tokens_path=tokens_path)

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
            self,
            prompt: str,
            stop=None,
            run_manager=None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

               # Instruction:
               {prompt}

               # Response:
               """
        return RWKVLLM.model(prompt)

    @property
    def _identifying_params(self):
        """Get the identifying parameters."""
        return {}


def get_llm(model_type, callbacks, model_path, tokenizer_path, model_n_ctx, repo_id, strategy):
    # Prepare the LLM
    match model_type:
        case "HuggingFace":
            from langchain import HuggingFaceHub
            llm = HuggingFaceHub(repo_id=repo_id)
        case "Neon":
            llm = NeonLLM()
        case "OpenAI":
            from langchain.llms import OpenAI
            llm = OpenAI()
        case "LlamaCpp":
            from langchain.llms import LlamaCpp
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks,
                           verbose=False, n_threads=16)
        case "GPT4All":
            from langchain.llms import GPT4All
            llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks,
                          verbose=False, n_threads=16)
        case "RWKV":
            RWKVLLM.init(model=model_path, tokens_path=tokenizer_path, strategy=strategy)
            llm = RWKVLLM()
        case "CTransformersMPT":
            from ctransformers.langchain import CTransformers
            llm = CTransformers(model=model_path, model_type='mpt')
        case "CTransformersGPT2":
            from ctransformers.langchain import CTransformers
            llm = CTransformers(model=model_path, model_type='gpt2')
        case "CTransformers":
            from ctransformers.langchain import CTransformers
            llm = CTransformers(model=model_path)
        case _default:
            print(f"Model {model_type} not supported!")
            exit(1)
    return llm


def llm_from_env():
    # generic model settings
    model_type = os.environ.get('MODEL_TYPE')
    model_path = os.environ.get('MODEL_PATH')

    # gpt4all / llama specific settings
    model_n_ctx = os.environ.get('MODEL_N_CTX')

    # HuggingFace specific settings
    repo_id = os.environ.get("REPO_ID")  # replaces model_path

    # RWKV specific settings
    tokenizer_path = os.environ.get('TOKENIZER_PATH')
    strategy = os.environ.get('STRATEGY')

    mute_stream = False
    callbacks = [] if mute_stream else [StreamingStdOutCallbackHandler()]
    llm = get_llm(model_type, callbacks, model_path, tokenizer_path, model_n_ctx, repo_id, strategy)
    return llm
