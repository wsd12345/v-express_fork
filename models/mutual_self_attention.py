import copy
from typing import List, Optional, Dict, Any

import torch
from torch import nn
from einops import rearrange

from third_part.V_Express.modules.attention import (
    BasicTransformerBlock,
    TemporalBasicTransformerBlock
)
from third_part.V_Express.modules.mutual_self_attention import (
    torch_dfs,
    ReferenceAttentionControl
)


class MyReferenceAttentionControl(ReferenceAttentionControl):
    def __init__(self, unet, mode, *args, **kwargs):
        super().__init__(unet, mode, *args, **kwargs)
        self.mode = mode
        if mode == "write":
            self.module_type = BasicTransformerBlock
        elif mode == "read":
            self.module_type = TemporalBasicTransformerBlock
        else:
            raise ValueError("mode={mode}. support ('read', 'write')")

    def register_reference_hooks(
            self,
            mode,
            do_classifier_free_guidance,
            attention_auto_machine_weight,
            gn_auto_machine_weight,
            style_fidelity,
            reference_attn,
            reference_adain,
            dtype=torch.float16,
            batch_size=1,
            num_images_per_prompt=1,
            device=torch.device("cpu"),
            fusion_blocks="midup",
    ):

        MODE = mode
        do_classifier_free_guidance = do_classifier_free_guidance

        num_images_per_prompt = num_images_per_prompt
        reference_attention_weight = self.reference_attention_weight
        audio_attention_weight = self.audio_attention_weight
        if do_classifier_free_guidance:
            uc_mask = (
                torch.Tensor(
                    [1] * batch_size * num_images_per_prompt * 16
                    + [0] * batch_size * num_images_per_prompt * 16
                )
                .to(device)
                .bool()
            )
        else:
            uc_mask = (
                torch.Tensor([0] * batch_size * num_images_per_prompt * 2)
                .to(device)
                .bool()
            )

        def hacked_basic_transformer_inner_forward(
                self,
                hidden_states: torch.FloatTensor,
                attention_mask: Optional[torch.FloatTensor] = None,
                encoder_hidden_states: Optional[torch.FloatTensor] = None,
                encoder_attention_mask: Optional[torch.FloatTensor] = None,
                timestep: Optional[torch.LongTensor] = None,
                cross_attention_kwargs: Dict[str, Any] = None,
                class_labels: Optional[torch.LongTensor] = None,
                video_length=None,
        ):
            if self.use_ada_layer_norm:  # False
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                (
                    norm_hidden_states,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                ) = self.norm1(
                    hidden_states,
                    timestep,
                    class_labels,
                    hidden_dtype=hidden_states.dtype,
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            # 1. Self-Attention
            # self.only_cross_attention = False
            cross_attention_kwargs = (
                cross_attention_kwargs if cross_attention_kwargs is not None else {}
            )
            if self.only_cross_attention:
                attn_output = self.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states
                    if self.only_cross_attention
                    else None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
            else:
                if MODE == "write":
                    attn_output = self.attn1(
                        norm_hidden_states,
                        encoder_hidden_states=encoder_hidden_states
                        if self.only_cross_attention
                        else None,
                        attention_mask=attention_mask,
                        **cross_attention_kwargs,
                    )

                    if self.use_ada_layer_norm_zero:
                        attn_output = gate_msa.unsqueeze(1) * attn_output
                    hidden_states = attn_output + hidden_states

                    if self.attn2 is not None:
                        norm_hidden_states = (
                            self.norm2(hidden_states, timestep)
                            if self.use_ada_layer_norm
                            else self.norm2(hidden_states)
                        )
                        self.bank.append(norm_hidden_states.clone())

                        # 2. Cross-Attention
                        attn_output = self.attn2(
                            norm_hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            attention_mask=encoder_attention_mask,
                            **cross_attention_kwargs,
                        )
                        hidden_states = attn_output + hidden_states

                if MODE == "read":
                    hidden_states = (
                            self.attn1(
                                norm_hidden_states,
                                encoder_hidden_states=norm_hidden_states,
                                attention_mask=attention_mask,
                            )
                            + hidden_states
                    )

                    if self.use_ada_layer_norm:  # False
                        norm_hidden_states = self.norm1_5(hidden_states, timestep)
                    elif self.use_ada_layer_norm_zero:
                        (
                            norm_hidden_states,
                            gate_msa,
                            shift_mlp,
                            scale_mlp,
                            gate_mlp,
                        ) = self.norm1_5(
                            hidden_states,
                            timestep,
                            class_labels,
                            hidden_dtype=hidden_states.dtype,
                        )
                    else:
                        norm_hidden_states = self.norm1_5(hidden_states)

                    bank_fea = []
                    # for d in self.bank:
                    #     if len(d.shape) == 3:
                    #         d = d.unsqueeze(1).repeat(1, video_length, 1, 1)
                    #      bank_fea.append(rearrange(d, "b t l c -> (b t) l c"))

                    num_d0 = norm_hidden_states.shape[0] // video_length
                    for d in self.bank:
                        if d.dim() == 3:  # b=2, l, c
                            if d.shape[0] == num_d0:
                                d = d.unsqueeze(1).repeat(1, video_length, 1, 1)
                            else:
                                if d.shape[0] % num_d0:
                                    raise Exception(" d.shape[0] % 2 != 0")
                                if not torch.all(d[:d.shape[0] // num_d0, ...] == 0):
                                    raise Exception(f"d[:d.shape[0]//{num_d0}, ...] != 0")
                                d = d.unsqueeze(1)
                        bank_fea.append(rearrange(d, "b t l c -> (b t) l c"))

                    attn_hidden_states = self.attn1_5(
                        norm_hidden_states,
                        encoder_hidden_states=bank_fea[0],
                        attention_mask=attention_mask,
                    )

                    if reference_attention_weight != 1.:
                        attn_hidden_states *= reference_attention_weight

                    hidden_states = (attn_hidden_states + hidden_states)

                    # self.bank.clear()
                    if self.attn2 is not None:
                        # Cross-Attention
                        norm_hidden_states = (
                            self.norm2(hidden_states, timestep)
                            if self.use_ada_layer_norm
                            else self.norm2(hidden_states)
                        )

                        attn_hidden_states = self.attn2(
                            norm_hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            attention_mask=attention_mask,
                        )

                        if audio_attention_weight != 1.:
                            attn_hidden_states *= audio_attention_weight

                        hidden_states = (attn_hidden_states + hidden_states)

                    # Feed-forward
                    hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

                    # Temporal-Attention
                    if self.unet_use_temporal_attention:
                        d = hidden_states.shape[1]
                        hidden_states = rearrange(
                            hidden_states, "(b f) d c -> (b d) f c", f=video_length
                        )
                        norm_hidden_states = (
                            self.norm_temp(hidden_states, timestep)
                            if self.use_ada_layer_norm
                            else self.norm_temp(hidden_states)
                        )
                        hidden_states = (
                                self.attn_temp(norm_hidden_states) + hidden_states
                        )
                        hidden_states = rearrange(
                            hidden_states, "(b d) f c -> (b f) d c", d=d
                        )

                    return hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = (
                        norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
                )

            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            hidden_states = ff_output + hidden_states

            return hidden_states

        if self.reference_attn:
            if self.fusion_blocks == "midup":
                attn_modules = [
                    module
                    for module in (
                            torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks)
                    )
                    if isinstance(module, BasicTransformerBlock)
                       or isinstance(module, TemporalBasicTransformerBlock)
                ]
            elif self.fusion_blocks == "full":
                attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, BasicTransformerBlock)
                       or isinstance(module, TemporalBasicTransformerBlock)
                ]
            attn_modules = sorted(
                attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )

            for i, module in enumerate(attn_modules):
                module._original_inner_forward = module.forward
                if isinstance(module, BasicTransformerBlock):
                    module.forward = hacked_basic_transformer_inner_forward.__get__(
                        module, BasicTransformerBlock
                    )
                if isinstance(module, TemporalBasicTransformerBlock):
                    module.forward = hacked_basic_transformer_inner_forward.__get__(
                        module, TemporalBasicTransformerBlock
                    )

                module.bank = []
                module.attn_weight = float(i) / float(len(attn_modules))

    def update(
            self,
            writer,
            do_classifier_free_guidance=True,
            dtype=torch.float16,
            device=torch.device("cpu"),
            writer_index=None,
    ):
        if not self.reference_attn:
            return
        if isinstance(writer, type(self)):
            self.__update_writer_self(writer, do_classifier_free_guidance, dtype)
        elif isinstance(writer, list):
            self.__update_writer_list(writer, do_classifier_free_guidance, dtype, device, writer_index)
        else:
            raise NotImplementedError(f"writer IS {type(writer)}")

    def get_attn_modules(self) -> List[nn.Module]:
        if self.fusion_blocks == "midup":
            attn_modules = [
                module
                for module in (torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks))
                if isinstance(module, self.module_type)
            ]

        elif self.fusion_blocks == "full":
            attn_modules = [
                module
                for module in torch_dfs(self.unet)
                if isinstance(module, self.module_type)
            ]

        else:
            raise ValueError(f"fusion_blocks = {self.fusion_blocks} not found. Support 'midup' and 'full'")
        attn_modules = sorted(
            attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
        )
        return attn_modules

    def get_attn_modules_parameter(self) -> List[List[torch.Tensor]]:
        attn_modules = self.get_attn_modules()
        res = []
        for attn in attn_modules:
            res.append([v.clone() for v in attn.bank])
            attn.bank.clear()
        return res

    def __update_writer_self(self,
                             writer,
                             do_classifier_free_guidance=True,
                             dtype=torch.float16,
                             ):
        reader_attn_modules = self.get_attn_modules()
        writer_attn_modules = writer.get_attn_modules()

        for r, w in zip(reader_attn_modules, writer_attn_modules):
            if do_classifier_free_guidance:
                r.bank = [torch.cat([torch.zeros_like(v), v]).to(dtype) for v in w.bank]
            else:
                r.bank = [v.clone().to(dtype) for v in w.bank]

    def __update_writer_list(self,
                             writer,
                             do_classifier_free_guidance=True,
                             dtype=torch.float16,
                             device=torch.device("cpu"),
                             writer_index=None,
                             ):
        reader_attn_modules = self.get_attn_modules()

        for r, w in zip(reader_attn_modules, writer):
            if do_classifier_free_guidance:
                if writer_index:
                    r.bank = [torch.cat([torch.zeros_like(v[writer_index]), v[writer_index]]).to(dtype).to(device)
                              for v in w]
                else:
                    r.bank = [torch.cat([torch.zeros_like(v), v]).to(dtype).to(device) for v in w]
            else:
                r.bank = [v.clone().to(dtype) for v in w.bank]
