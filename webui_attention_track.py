# How to use
# uvicorn webui_attention_track:app --reload --host 0.0.0.0 --port 8089

import asyncio
import copy
import http
from http import HTTPStatus
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import json
from typing import Optional, Union
import fastapi
import torch
import torch.nn.functional as torchF
import transformers
from fastapi.responses import (HTMLResponse, JSONResponse, RedirectResponse,
                               StreamingResponse)
from fastapi import HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import (AutoModelForCausalLM, AutoTokenizer, DynamicCache,
                          GenerationConfig, PreTrainedModel, PreTrainedTokenizer, AutoConfig)
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList, EosTokenCriteria

from transformers.generation.utils import TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

import uuid

@dataclass
class ConversationItem:
    role: str = field(default="user")
    content: str = field(default="")

@dataclass
class GenerateArgs:
    conversation: list[ConversationItem] = field(default_factory=list)
    max_new_tokens: int = 4096
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 10
    do_sample: bool = True

    # for for tokeinzer
    add_generation_prompt: bool = True
    # for attention
    output_attention: bool = False
    output_attention_layer_idx: int = 0
    output_attention_head_idx: int = 0

    # in ms, sleep after each generation step
    pause_duration: Optional[float] = None

    def update_output_attention(self, output_attention: Optional[bool], output_attention_layer_idx: Optional[int], output_attention_head_idx: Optional[int]):
        """
        Just set the output_attention, output_attention_layer_index, output_attention_head_index
        No any check
        """
        if output_attention is not None:
            self.output_attention = output_attention
        if output_attention_layer_idx is not None:
            self.output_attention_layer_idx = output_attention_layer_idx
        if output_attention_head_idx is not None:
            self.output_attention_head_idx = output_attention_head_idx
        return self

@dataclass
class ChatStreamItem(GenerateArgs):
    input_ids: Optional[torch.Tensor] = field(default=None)
    forward_kwargs: dict = field(default_factory=dict)

    logits_processor: Optional[LogitsProcessorList] = field(default=None)
    stop_criteria: Optional[StoppingCriteriaList] = field(default=None)

    is_prefill: bool = True
    eos_hitted: bool = False

    async_event: asyncio.Event = field(default_factory=asyncio.Event)
    is_paused: bool = False
    
    is_force_stop: bool = False
    
    def is_generating(self):
        if self.is_force_stop:
            return False
        if self.eos_hitted:
            return False
        if self.generation_index >= self.max_new_tokens:
            return False
        return True
    
    async def wait(self):
        if self.is_paused:
            await self.async_event.wait()
        elif (self.pause_duration is not None) and (not self.is_prefill):
            # escape if is prefill
            await asyncio.sleep(self.pause_duration / 1000)
        return not self.is_force_stop

    @property
    def generation_index(self):
        return self.input_ids.shape[-1] if self.input_ids is not None else 0
    
    def free_resources(self):
        del self.input_ids
        self.input_ids = None
        del self.forward_kwargs
        self.forward_kwargs = dict()
        torch.cuda.empty_cache()

    def update_output_attention(self, output_attention, output_attention_layer_index, output_attention_head_index):
        super().update_output_attention(output_attention, output_attention_layer_index, output_attention_head_index)
        self.forward_kwargs["output_attentions"] = self.output_attention
        return self

class GlobalVars:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device = torch.device("cuda:0"),
    ):
        self.model_name = model_name
        self.divice = device

        self.model = None
        self.tokenizer = None
        self.generation_config = None
        self.model_config = None

        self.forward_stream_cache = dict() # type: dict[str, ChatStreamItem]
    
    def __del__(self):
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        self.model_config = None

    @property
    def num_hidden_layers(self):
        self.load_config()
        return self.model_config.num_hidden_layers

    @property
    def num_attention_heads(self):
        self.load_config()
        return self.model_config.num_attention_heads

    def load_config(self):
        if self.model_config is None:
            self.model_config = AutoConfig.from_pretrained(self.model_name)
        return self.model_config

    def load_model(self) -> PreTrainedModel:
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager", # to support attenton weights
            ).to(self.divice).eval()
            self.model_config = self.model.config
        return self.model

    def load_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self.tokenizer

    def load_generation_config(self):
        if self.generation_config is None:
            tokenizer = self.load_tokenizer()
            self.generation_config = GenerationConfig.from_pretrained(
                self.model_name,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                max_length=tokenizer.model_max_length,
            )
        model = self.load_model()

        model._prepare_special_tokens(
            self.generation_config,
            kwargs_has_attention_mask=True,
            device=self.divice,
        )

        return self.generation_config
    
    def copy_generation_config(self, args: Optional[GenerateArgs] = None):
        generation_config = copy.deepcopy(self.load_generation_config())
        if args is None:
            return generation_config
        generation_config.max_new_tokens = args.max_new_tokens
        generation_config.temperature = args.temperature
        generation_config.top_p = args.top_p
        generation_config.do_sample = args.do_sample
        generation_config.top_k = args.top_k
        return generation_config
    
    def check_valid_generate_args(self, args: GenerateArgs):
        return self.check_valid_output_attention_args(
            args.output_attention_layer_idx,
            args.output_attention_head_idx,
        )

    def check_uuid(self, id: str):
        if id not in self.forward_stream_cache:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail="uuid not found")
        return True
    
    def check_valid_output_attention_args(
        self,
        output_attention_layer_index: Optional[int] = None,
        output_attention_head_index: Optional[int] = None,
    ):
        if output_attention_layer_index is not None:
            if output_attention_layer_index < 0:
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST,
                    detail="output_attention_layer_index must be non-negative")
            if output_attention_layer_index >= self.num_hidden_layers:
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST,
                    detail=f"output_attention_layer_index must be less than {self.num_hidden_layers}")

        if output_attention_head_index is not None:
            if output_attention_head_index < 0:
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST,
                    detail="output_attention_head_index must be non-negative")

            if output_attention_head_index >= self.num_attention_heads:
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST,
                    detail=f"output_attention_head_index must be less than {self.num_attention_heads}")

        return True


    # forward stream cache
    def get_forward_stream_item(self, id: str) -> ChatStreamItem:
        if id not in self.forward_stream_cache:
            raise HTTPException(
                HTTPStatus.BAD_REQUEST,
                detail="uuid not found")
        return self.forward_stream_cache[id]
    
    def remove_forward_stream_item(self, id: str):
        if id not in self.forward_stream_cache:
            return
        del self.forward_stream_cache[id]
    
    def new_forward_stream(self, args: GenerateArgs, add_to_cache: bool=False): 
        item = ChatStreamItem(**vars(args))
        
        model = self.load_model()
        tokenizer = self.load_tokenizer()
        inputs = tokenizer.apply_chat_template(
            args.conversation,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=args.add_generation_prompt,
        ).to(self.divice)

        input_ids = inputs["input_ids"] # type: torch.Tensor
        attention_mask = inputs["attention_mask"] # type: torch.Tensor

        generation_config = self.copy_generation_config(args)
        logits_processor = model._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids.shape[-1],
            encoder_input_ids=input_ids,
            prefix_allowed_tokens_fn=None,
            logits_processor=[],
            device=gvars.divice,
        )

        stop_criteria = model._get_stopping_criteria(
            generation_config=generation_config,
            stopping_criteria=[],
            tokenizer=tokenizer,
        )

        item.input_ids = input_ids
        item.logits_processor = logits_processor
        item.stop_criteria = stop_criteria
        
        item.forward_kwargs = dict(
            attention_mask=attention_mask,
            output_attentions=args.output_attention,
            past_key_values=DynamicCache(),
            cache_position=torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device),
        )

        if add_to_cache:
            _uuid = uuid.uuid4().hex
            self.forward_stream_cache[_uuid] = item
            return item, _uuid

        return item

gvars = None

@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    print("lifespan start")
    global gvars
    gvars = GlobalVars(device=torch.device("cuda:0"))
    yield
    del gvars
    gvars = None
    print("lifespan end")

app = fastapi.FastAPI(debug=True, lifespan=lifespan)

# static 
app.mount("/static", StaticFiles(directory="./webui/attention_track/static"), name="static")
templates = Jinja2Templates(directory="./webui/attention_track/template")

@app.get("/", response_class=HTMLResponse)
async def index(request: fastapi.Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "model_name": gvars.model_name,
            "num_hidden_layers": gvars.num_hidden_layers,
            "num_attention_heads": gvars.num_attention_heads,
        },
    )


@app.post("/api/chat")
async def chat(args: GenerateArgs):
    model = gvars.load_model()
    tokenizer = gvars.load_tokenizer()
    inputs = tokenizer.apply_chat_template(
        args.conversation,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=args.add_generation_prompt,
    ).to(gvars.divice)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        do_sample=args.do_sample,
    )

    generated = outputs[:, input_ids.shape[1]:]
    generated_text = tokenizer.batch_decode(generated, skip_special_tokens=True)

    return JSONResponse(content={
        "status": "success",
        "content": generated_text,
    })


@app.post("/api/apply_chat_template")
async def apply_chat_template(args: GenerateArgs):
    tokenizer = gvars.load_tokenizer()
    inputs = tokenizer.apply_chat_template(
        args.conversation,
        tokenize=False,
        add_generation_prompt=args.add_generation_prompt,
    )

    tokens = tokenizer(
        inputs,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]


    token_split_txt = tokenizer.batch_decode(tokens, skip_special_tokens=False)
    
    return JSONResponse(content={
        "status": "success",
        "content": inputs,
        "tokens": tokens,
        "token_split_txt": token_split_txt,
    })

@app.get("/api/model_info")
async def model_info():
    return JSONResponse(content=gvars.model_config.to_dict())

@app.get("/api/model_num_layers")
async def model_num_layers():
    return JSONResponse(content={
        "status": "success",
        "num_layers": gvars.num_hidden_layers,
    })

@app.get("/api/model_num_heads")
async def model_num_heads():
    return JSONResponse(content={
        "status": "success",
        "num_heads": gvars.num_attention_heads,
    })

@app.post("/sse/chat_stream")
async def sse_chat_stream(args: GenerateArgs):

    def stream_output(src: dict):
        return f"{json.dumps(src, ensure_ascii=False)}\n"

    async def stream(
        _uuid: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        sitem: ChatStreamItem,
    ):
        yield stream_output({
            "uuid": _uuid,
            "status": "start",
        })

        input_ids = sitem.input_ids
        flush_length = 0
        while sitem.is_generating():

            model_inputs = model.prepare_inputs_for_generation(
                input_ids,
                **sitem.forward_kwargs,
            )

            with torch.no_grad():
                outputs = model.forward(
                    **model_inputs,
                    return_dict=True,
                ) # type: dict

            sitem.forward_kwargs = model._update_model_kwargs_for_generation(
                outputs=outputs,
                model_kwargs=sitem.forward_kwargs,
                is_encoder_decoder=model.config.is_encoder_decoder,
            )

            next_token_logits = outputs.logits[:, -1, :].clone().float()
            next_token_logits = next_token_logits.to(gvars.divice)

            next_token_scores = sitem.logits_processor(input_ids, next_token_logits)

            if sitem.do_sample:
                probs = torchF.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)
            
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            sitem.input_ids = input_ids

            if sitem.is_prefill:
                sitem.is_prefill = False
                _str_tokens = tokenizer.batch_decode(
                    input_ids[0].tolist(),
                    skip_special_tokens=False,
                )
            else:
                _str_tokens = tokenizer.batch_decode(
                    input_ids[0, -2:].tolist(),
                    skip_special_tokens=False,
                )
            
            _q_len = len(_str_tokens) - 1

            if sitem.output_attention:
                attentions = outputs.attentions

                _layer_idx = sitem.output_attention_layer_idx
                _head_idx = sitem.output_attention_head_idx

                _resp_json = {
                    "status": "generating",
                    "layer_idx": _layer_idx,
                    "head_idx": _head_idx,
                }

                # bsz = 1 for sure
                # shape (q_len, k_len)
                picked_attn_head = attentions[_layer_idx][0, _head_idx]

                for q in range(_q_len):
                    flush_length += 1
                    _resp_json.update({
                        "current": _str_tokens[q],
                        "attention": picked_attn_head[q, 0:flush_length].tolist(),
                        "next": _str_tokens[q+1],
                    })
                    yield stream_output(_resp_json)
                    if not await sitem.wait():
                        break
            else:
                for q in range(_q_len):
                    flush_length += 1
                    _resp_json = {
                        "status": "generating",
                        "current": _str_tokens[q],
                        "next": _str_tokens[q+1],
                    }
                    yield stream_output(_resp_json)
                    if not await sitem.wait():
                        break

            del outputs

            stop_check = sitem.stop_criteria(input_ids, None)
            if torch.all(stop_check):
                sitem.eos_hitted = True
                # flush_length += 1
                break
        
        # end while
        # delete resources
        yield stream_output({
            "status": "end",
            "current": _str_tokens[-1],
        })
        gvars.remove_forward_stream_item(_uuid)
    
    # check valid input
    gvars.check_valid_generate_args(args)

    model = gvars.load_model()
    tokenizer = gvars.load_tokenizer()

    csi, _uuid = gvars.new_forward_stream(args, add_to_cache=True)

    return StreamingResponse(
        stream(
            _uuid=_uuid,
            model=model,
            tokenizer=tokenizer,
            sitem=csi,
        ),
        media_type="text/event-stream",
    )

@app.get("/sse/chat_stream/setting")
async def sse_chat_stream_setting(
    uuid: str = fastapi.Query(...),
    output_attention: Optional[bool] = None,
    output_attention_layer_idx: Optional[int] = None,
    output_attention_head_idx: Optional[int] = None,
):
    gvars.check_valid_output_attention_args(
        output_attention_layer_idx,
        output_attention_head_idx,
    )

    sitem = gvars.get_forward_stream_item(uuid)
    sitem.update_output_attention(
        output_attention,
        output_attention_layer_idx,
        output_attention_head_idx,
    )
    return JSONResponse(content={
        "status": "success",
    })

@app.get("/sse/chat_stream/pause")
async def sse_chat_stream_pause(uuid: str, pause: bool):
    sitem = gvars.get_forward_stream_item(uuid)
    async with asyncio.Lock():
        if sitem.is_paused:
            # pausing now
            if pause:
                # set to pause
                # just clear the state
                sitem.async_event.clear()
            else:
                # set to resume
                sitem.is_paused = False
                sitem.async_event.set()
                sitem.async_event.clear()
        else:
            sitem.async_event.clear()
            # running 
            if pause:
                # set to pause
                sitem.is_paused = True
            else:
                # do nothing
                pass

    return JSONResponse(content={
        "status": "success",
    })

@app.get("/sse/chat_stream/pause_duration")
async def sse_chat_stream_pause_duration(uuid: str, pause_duration: int):
    sitem = gvars.get_forward_stream_item(uuid)
    sitem.pause_duration = pause_duration
    return JSONResponse(content={
        "status": "success",
    })

@app.get("/sse/chat_stream/stop")
async def sse_chat_stream_stop(uuid: str):
    sitem = gvars.get_forward_stream_item(uuid)
    async with asyncio.Lock():
        if sitem.is_paused:
            sitem.is_paused = False
            sitem.async_event.set()
        sitem.async_event.clear()
        sitem.is_force_stop = True

    return JSONResponse(content={
        "status": "success",
    })
