import math

import numpy as np
from PIL import Image
from typing import List, Optional, Union, Callable, Iterator
from einops import rearrange
import torch
from itertools import zip_longest
from diffusers.utils.torch_utils import randn_tensor

from models.mutual_self_attention import MyReferenceAttentionControl
from third_part.V_Express.pipelines import VExpressPipeline
from third_part.V_Express.pipelines.context import get_context_scheduler
from third_part.V_Express.pipelines.v_express_pipeline import retrieve_timesteps


class MyVExpressPipeline(VExpressPipeline):

    def get_num_frame_context(self, context_schedule, context_frames, context_overlap, video_length, ):
        device = self._execution_device
        context_scheduler = get_context_scheduler(context_schedule)
        context_queue = list(
            context_scheduler(
                step=0,
                num_frames=video_length,
                context_size=context_frames,
                context_stride=1,
                context_overlap=context_overlap,
                closed_loop=False,
            )
        )

        num_frame_context = torch.zeros(video_length, device=device, dtype=torch.long)
        for context in context_queue:
            num_frame_context[context] += 1

        return context_queue, num_frame_context

    def get_ref_img_weight(self,
                           reference_images: List[Image.Image],
                           reference_control_writer: MyReferenceAttentionControl,
                           height: int, width: int,
                           context_frames: int = 24,
                           save_gpu_memory: bool = False,
                           ):
        # reference_images_latents = self.prepare_reference_latent(reference_images, height, width)
        # batch_size_ref_image = reference_images_latents.shape[0]
        # encoder_hidden_states = torch.zeros((batch_size_ref_image, 1, 768), dtype=self.dtype, device=self.device)
        # self.reference_net(
        #     reference_images_latents,
        #     timestep=0,
        #     encoder_hidden_states=encoder_hidden_states,
        #     return_dict=False,
        # )
        # reference_control_reader.update(reference_control_writer, do_classifier_free_guidance)
        # bs = 8 need 3.3G Memory on rtx3090(CUDA Version: 12.4) with no_grad
        bs_ref_img = max(2 ** int(pow(context_frames, 0.5)), 1)
        ref_param = []
        for i in range(math.ceil(len(reference_images) / bs_ref_img)):
            reference_images_latents = self.prepare_reference_latent(
                reference_images[bs_ref_img * i:bs_ref_img * (i + 1)], height, width)
            encoder_hidden_states = torch.zeros(
                (reference_images_latents.shape[0], 1, 768), dtype=self.dtype, device=self.device)
            self.reference_net(
                reference_images_latents,
                timestep=0,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )
            if not ref_param:
                ref_param = [[v.clone().cpu() for v in vs] for vs in
                             reference_control_writer.get_attn_modules_parameter()]
            else:
                for i, vs in enumerate(reference_control_writer.get_attn_modules_parameter()):
                    for v in vs:
                        ref_param[i].append(v.clone().cpu())
        if save_gpu_memory:
            del self.reference_net
        torch.cuda.empty_cache()

        return [[torch.concatenate(i)] for i in ref_param]

    @torch.no_grad()
    def __call__(
            self,
            reference_images,
            kps_images,
            audio_waveform,
            width,
            height,
            video_length,
            num_inference_steps,
            guidance_scale,
            strength=1.,
            num_images_per_prompt=1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,

            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            context_schedule="uniform",
            context_frames=24,
            context_overlap=4,
            reference_attention_weight=1.,
            audio_attention_weight=1.,
            num_pad_audio_frames=2,
            do_multi_devices_inference=False,
            save_gpu_memory=False,
            **kwargs,
    ):
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0
        batch_size = 1  # not use

        # Prepare timesteps
        timesteps = None
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)

        reference_control_writer = MyReferenceAttentionControl(
            self.reference_net,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="write",
            batch_size=batch_size,
            fusion_blocks="full",
        )
        reference_control_reader = MyReferenceAttentionControl(
            self.denoising_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="read",
            batch_size=batch_size,
            fusion_blocks="full",
            reference_attention_weight=reference_attention_weight,
            audio_attention_weight=audio_attention_weight,
        )

        num_channels_latents = self.denoising_unet.in_channels
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # kps
        if kps_images:
            kps_feature = self.prepare_kps_feature(kps_images, height, width, do_classifier_free_guidance)
            if save_gpu_memory:
                del self.v_kps_guider
            torch.cuda.empty_cache()

        # audio
        audio_embeddings = self.prepare_audio_embeddings(
            audio_waveform,
            video_length,
            num_pad_audio_frames,
            do_classifier_free_guidance,
        )
        if save_gpu_memory:
            del self.audio_processor, self.audio_encoder, self.audio_projection
        torch.cuda.empty_cache()

        # context index
        context_queue, num_frame_context = self.get_num_frame_context(
            self, context_schedule, context_frames,
            context_overlap, video_length, )

        # reference images
        reference_control_writer_weight = self.get_ref_img_weight(
            reference_images,
            reference_control_writer,
            height, width,
            context_frames,
            save_gpu_memory, )

        # random noise
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            width,
            height,
            video_length,
            self.dtype,
            torch.device('cpu'),
            generator,
        )

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                context_counter = torch.zeros(video_length, device=device, dtype=torch.long)
                noise_preds = [None] * video_length
                for context_idx, context in enumerate(context_queue):
                    # update reference images feature
                    reference_control_reader.update(
                        reference_control_writer_weight, do_classifier_free_guidance, writer_index=context)

                    if kps_images:
                        latent_kps_feature = kps_feature[:, :, context].to(device, self.dtype)
                    else:
                        latent_kps_feature = None

                    latent_audio_embeddings = audio_embeddings[:, context, ...]
                    _, _, num_tokens, dim = latent_audio_embeddings.shape
                    latent_audio_embeddings = latent_audio_embeddings.reshape(-1, num_tokens, dim)

                    input_latents = latents[:, :, context, ...].to(device)
                    input_latents = input_latents.repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
                    input_latents = self.scheduler.scale_model_input(input_latents, t)
                    noise_pred = self.denoising_unet(
                        input_latents,
                        t,
                        encoder_hidden_states=latent_audio_embeddings.reshape(-1, num_tokens, dim),
                        kps_features=latent_kps_feature,
                        return_dict=False,
                    )[0]
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    context_counter[context] += 1
                    noise_pred /= num_frame_context[context][None, None, :, None, None]
                    step_frame_ids = []
                    step_noise_preds = []
                    for latent_idx, frame_idx in enumerate(context):
                        if noise_preds[frame_idx] is None:
                            noise_preds[frame_idx] = noise_pred[:, :, latent_idx, ...]
                        else:
                            noise_preds[frame_idx] += noise_pred[:, :, latent_idx, ...]
                        if context_counter[frame_idx] == num_frame_context[frame_idx]:
                            step_frame_ids.append(frame_idx)
                            step_noise_preds.append(noise_preds[frame_idx])
                            noise_preds[frame_idx] = None
                    step_noise_preds = torch.stack(step_noise_preds, dim=2)
                    output_latents = self.scheduler.step(
                        step_noise_preds,
                        t,
                        latents[:, :, step_frame_ids, ...].to(device),
                        **extra_step_kwargs,
                    ).prev_sample
                    latents[:, :, step_frame_ids, ...] = output_latents.cpu()

                    progress_bar.set_description(
                        f'Denoising Step Index: {i + 1} / {len(timesteps)}, '
                        f'Context Index: {context_idx + 1} / {len(context_queue)}'
                    )

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        reference_control_reader.clear()
        reference_control_writer.clear()

        video_tensor = self.decode_latents(latents)
        return video_tensor


class AudioPipeline(MyVExpressPipeline):
    def __init__(self,
                 audio_processor,
                 audio_encoder,
                 audio_projection,
                 *args, **kwargs):
        self.register_modules(

            audio_processor=audio_processor,
            audio_encoder=audio_encoder,
            audio_projection=audio_projection,

        )

    @torch.no_grad()
    def prepare_audio_embeddings(self,
                                 audio_waveform, video_length, num_pad_audio_frames, do_classifier_free_guidance):
        audio_waveform = self.audio_processor(audio_waveform, return_tensors="pt", sampling_rate=16000)['input_values']
        audio_waveform = audio_waveform.to(self.device, self.dtype)
        audio_embeddings = self.audio_encoder(audio_waveform).last_hidden_state  # [1, num_embeds, d]

        audio_embeddings = torch.nn.functional.interpolate(
            audio_embeddings.permute(0, 2, 1),
            size=2 * video_length,
            mode='linear',
        )[0, :, :].permute(1, 0)  # [2*vid_len, dim]

        audio_embeddings = torch.cat([
            torch.zeros_like(audio_embeddings)[:2 * num_pad_audio_frames, :],
            audio_embeddings,
            torch.zeros_like(audio_embeddings)[:2 * num_pad_audio_frames, :],
        ], dim=0)  # [2*num_pad+2*vid_len+2*num_pad, dim]

        frame_audio_embeddings = []
        for frame_idx in range(video_length):
            start_sample = frame_idx
            end_sample = frame_idx + 2 * num_pad_audio_frames

            frame_audio_embedding = audio_embeddings[2 * start_sample:2 * (end_sample + 1), :]  # [2*num_pad+1, dim]
            frame_audio_embeddings.append(frame_audio_embedding)
        audio_embeddings = torch.stack(frame_audio_embeddings, dim=0)  # [vid_len, 2*num_pad+1, dim]

        audio_embeddings = self.audio_projection(audio_embeddings).unsqueeze(0)
        if do_classifier_free_guidance:
            uc_audio_embeddings = torch.zeros_like(audio_embeddings)
            audio_embeddings = torch.cat([uc_audio_embeddings, audio_embeddings], dim=0)
        return audio_embeddings

    def __call__(self, audio_waveforms, video_lengths,
                 num_pad_audio_frames, do_classifier_free_guidance,
                 *args, **kwargs):

        for audio, l in zip(audio_waveforms, video_lengths):
            yield self.prepare_audio_embeddings(
                audio, l, num_pad_audio_frames, do_classifier_free_guidance
            )


class ReferencePipeline(MyVExpressPipeline):
    @torch.no_grad()
    def __call__(self,
                 reference_images,
                 kps_images,
                 width,
                 height,
                 context_frames,
                 guidance_scale,
                 *args, **kwargs):
        do_classifier_free_guidance = guidance_scale > 1.0

        reference_control_writer = MyReferenceAttentionControl(
            self.reference_net,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="write",
            fusion_blocks="full",
        )

        if kps_images:
            kps_feature = self.prepare_kps_feature(kps_images, height, width, do_classifier_free_guidance)
            torch.cuda.empty_cache()
        else:
            kps_feature = None
        # st = time.monotonic()
        reference_control_writer_weight = self.get_ref_img_weight(
            reference_images,
            reference_control_writer,
            height, width,
            context_frames,
        )
        # print(time.monotonic()-st, len(reference_images))
        torch.cuda.empty_cache()
        return reference_control_writer_weight, kps_feature


class LipPipeline(MyVExpressPipeline):
    @torch.no_grad()
    def decode_latents(self, latents_obj) -> Iterator[np.ndarray]:
        for i in range(len(latents_obj) + 1):

            latents = 1 / 0.18215 * latents_obj[i]
            latents = rearrange(latents, "b c f h w -> (b f) c h w")

            for frame_idx in range(latents.shape[0]):
                image = self.vae.decode(latents[frame_idx: frame_idx + 1].to(self.vae.device)).sample
                image = (image / 2 + 0.5).clamp(0, 1)
                yield image.squeeze(0).cpu().float().permute(1, 2, 0).numpy()

    @torch.no_grad()
    def __call__(
            self,
            audio_embeddings,
            face_feature_obj,
            latents_obj,
            video_length,
            num_inference_steps,
            guidance_scale,
            strength=1.,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            output_type: Optional[str] = "tensor",
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            context_schedule="uniform",
            context_frames=24,
            context_overlap=4,
            reference_attention_weight=1.,
            audio_attention_weight=1.,
            do_multi_devices_inference=False,
            save_gpu_memory=False,
            **kwargs,
    ):
        contest_bs = context_frames - context_overlap
        if len(face_feature_obj) * contest_bs + face_feature_obj.pre_length < (len(latents_obj) + 1) * contest_bs:
            raise Exception("number of face_feature_obj < number of latents_obj")
        device = self._execution_device
        audio_embeddings = audio_embeddings.to(device)

        do_classifier_free_guidance = guidance_scale > 1.0

        # Prepare timesteps
        timesteps = None
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)

        reference_control_reader = MyReferenceAttentionControl(
            self.denoising_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="read",
            batch_size=1,
            fusion_blocks="full",
            reference_attention_weight=reference_attention_weight,
            audio_attention_weight=audio_attention_weight,
        )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # context index
        context_queue, num_frame_context = self.get_num_frame_context(
            context_schedule, context_frames,
            context_overlap, video_length, )

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        for i, t in enumerate(timesteps):
            context_counter = torch.zeros(video_length, device=device, dtype=torch.long)
            noise_preds = [None] * video_length

            myzip = zip_longest if i == len(timesteps) - 1 else zip
            print(i, myzip)
            for context_idx, (context, face_feature, latents) in enumerate(
                    myzip(context_queue, face_feature_obj, latents_obj)):
                if context is None:
                    break
                # update reference images feature
                reference_control_reader.update(
                    face_feature["reference_control_writer_weight"],
                    do_classifier_free_guidance, device=device)

                if face_feature["kps_feature"] is not None:
                    latent_kps_feature = face_feature["kps_feature"].to(device, self.dtype)
                else:
                    latent_kps_feature = None

                latent_audio_embeddings = audio_embeddings[:, context, ...]
                _, _, num_tokens, dim = latent_audio_embeddings.shape
                latent_audio_embeddings = latent_audio_embeddings.reshape(-1, num_tokens, dim)

                input_latents = latents.to(device)
                input_latents = input_latents.repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
                input_latents = self.scheduler.scale_model_input(input_latents, t)
                noise_pred = self.denoising_unet(
                    input_latents,
                    t,
                    encoder_hidden_states=latent_audio_embeddings.reshape(-1, num_tokens, dim),
                    kps_features=latent_kps_feature,
                    return_dict=False,
                )[0]
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                context_counter[context] += 1
                noise_pred /= num_frame_context[context][None, None, :, None, None]
                step_frame_ids = []
                step_noise_preds = []
                for latent_idx, frame_idx in enumerate(context):
                    if noise_preds[frame_idx] is None:
                        noise_preds[frame_idx] = noise_pred[:, :, latent_idx, ...]
                    else:
                        noise_preds[frame_idx] += noise_pred[:, :, latent_idx, ...]
                    if context_counter[frame_idx] == num_frame_context[frame_idx]:
                        step_frame_ids.append(frame_idx - context[0])
                        step_noise_preds.append(noise_preds[frame_idx])
                        noise_preds[frame_idx] = None
                step_noise_preds = torch.stack(step_noise_preds, dim=2)
                output_latents = self.scheduler.step(
                    step_noise_preds,
                    t,
                    latents[:, :, step_frame_ids, ...].to(device),
                    **extra_step_kwargs,
                ).prev_sample
                if context_idx != len(latents_obj) - 1:
                    latents_obj.save(output_latents.cpu(), context_idx)
                else:
                    latents_obj.save(output_latents[:, :, :-context_overlap].cpu(), context_idx)
                    latents_obj.save(output_latents[:, :, -context_overlap:].cpu(), context_idx + 1)

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):

                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, latents)

        reference_control_reader.clear()
        # video_tensor = self.decode_latents(latents)
        return None
