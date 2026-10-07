"""Reuse native EXL3 JSON filters in append-only VLM sessions."""
from rocm_tools.exl3_server.schema import normalize_response_format,_llguidance_schema,validate_response_format_output


class SessionJSONConstraint:
    def __init__(self,runtime,response_format,thinking=False):
        self.runtime=runtime
        self.format=normalize_response_format(response_format)
        if self.format is None:raise ValueError('A JSON response format is required')
        if thinking and getattr(runtime,'reasoning_end_id',None) is None:
            raise ValueError('Constrained thinking needs a recognized native reasoning boundary')
        from exllamav3.generator.filter import LLGuidanceFilter
        # Thought and native boundary tokens are handled by the session loop;
        # this matcher consumes only the generated final JSON, never prompt IDs.
        self.filter=LLGuidanceFilter(runtime.tokenizer,trigger_token=None,eos_after_completed=True,
                                     json_schema=_llguidance_schema(self.format['schema']))
        self.tokens=[]

    def mask(self,logits):
        t=self.runtime.torch
        mask=self.filter.get_next_logit_mask().to(logits.device)
        if mask.dtype==t.int32:
            if logits.is_cuda:
                from exllamav3.ext import exllamav3_ext as ext
                output=t.empty_like(logits).view(1,-1)
                ext.apply_logit_bitmask(logits.contiguous().view(1,-1),output,mask.contiguous())
                result=output.flatten()
            else:
                indices=t.arange(logits.numel(),device=logits.device)
                words=mask.flatten()
                allowed=((words[(indices//32).clamp_max(words.numel()-1)]>>(indices%32))&1).bool()
                allowed&=indices<words.numel()*32
                result=logits.masked_fill(~allowed,float('-inf'))
        else:
            additive=mask.flatten()
            if additive.numel()<logits.numel():
                additive=t.nn.functional.pad(additive,(0,logits.numel()-additive.numel()),value=float('-inf'))
            result=logits+additive[:logits.numel()]
        if not bool(t.isfinite(result).any()):raise ValueError('JSON grammar has no finite allowed next token')
        return result

    def accept(self,token):
        self.tokens.append(token)
        return self.filter.feed(token)

    def finish(self):
        text=self.runtime.tokenizer.decode(self.runtime.torch.tensor(self.tokens,dtype=self.runtime.torch.long),decode_special_tokens=True)
        validate_response_format_output(text,self.format)
        return {'type':self.format['type'],'name':self.format.get('name'),
                'constrained_tokens':len(self.tokens),'validated':True,'backend':'native_llguidance'}


def build_session_constraint(runtime,response_format,thinking=False):
    if normalize_response_format(response_format) is None:return None
    return SessionJSONConstraint(runtime,response_format,thinking)
