class Script(scripts.Script):

    def __init__(self):
        self.hooked = False

    def title(self):
        return "Tiled VAE"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        tab = 't2i' if not is_img2img else 'i2i'
        def uid(name):
            return f'tiledvae-{tab}-{name}'
        with gr.Accordion('Tiled VAE', open=False):
            with gr.Row() as tab_enable:
                enabled = gr.Checkbox(label='Enable Tiled VAE', value=False, elem_id=uid('enable'))
                vae_to_gpu = gr.Checkbox(label='Move VAE to GPU (if possible)', value=True, elem_id=uid('vae2gpu'))

            gr.HTML('<p style="margin-bottom:0.8em"> Recommended to set tile sizes as large as possible before got CUDA error: out of memory. </p>')
            with gr.Row() as tab_size:
                encoder_tile_size = gr.Slider(label='Encoder Tile Size', minimum=256, maximum=4096, step=16, value=get_rcmd_enc_tsize(), elem_id=uid('enc-size'))
                decoder_tile_size = gr.Slider(label='Decoder Tile Size', minimum=48,  maximum=512,  step=16, value=get_rcmd_dec_tsize(), elem_id=uid('dec-size'))
                reset = gr.Button(value='↻ Reset', variant='tool')
                reset.click(fn=lambda: [get_rcmd_enc_tsize(), get_rcmd_dec_tsize()], outputs=[encoder_tile_size, decoder_tile_size], show_progress=False)

            with gr.Row() as tab_param:
                fast_encoder = gr.Checkbox(label='Fast Encoder', value=True, elem_id=uid('fastenc'))
                color_fix    = gr.Checkbox(label='Fast Encoder Color Fix', value=False, visible=True, elem_id=uid('fastenc-colorfix'))
                fast_decoder = gr.Checkbox(label='Fast Decoder', value=True, elem_id=uid('fastdec'))

                fast_encoder.change(fn=gr_show, inputs=fast_encoder, outputs=color_fix, show_progress=False)

        return [
            enabled, 
            encoder_tile_size, decoder_tile_size,
            vae_to_gpu, fast_decoder, fast_encoder, color_fix, 
        ]

    def process(self, p:Processing, 
            enabled:bool, 
            encoder_tile_size:int, decoder_tile_size:int, 
            vae_to_gpu:bool, fast_decoder:bool, fast_encoder:bool, color_fix:bool
        ):
        
        # for shorthand
        vae = p.sd_model.first_stage_model
        encoder = vae.encoder
        decoder = vae.decoder

        # undo hijack if disabled (in cases last time crashed)
        if not enabled:
            if self.hooked:
                if isinstance(encoder.forward, VAEHook):
                    encoder.forward.net = None
                    encoder.forward = encoder.original_forward
                if isinstance(decoder.forward, VAEHook):
                    decoder.forward.net = None
                    decoder.forward = decoder.original_forward
                self.hooked = False
            return

        if devices.get_optimal_device_name().startswith('cuda') and vae.device == devices.cpu and not vae_to_gpu:
            print("[Tiled VAE] warn: VAE is not on GPU, check 'Move VAE to GPU' if possible.")

        # do hijack
        kwargs = {
            'fast_decoder': fast_decoder, 
            'fast_encoder': fast_encoder, 
            'color_fix':    color_fix, 
            'to_gpu':       vae_to_gpu,
        }

        # save original forward (only once)
        if not hasattr(encoder, 'original_forward'): setattr(encoder, 'original_forward', encoder.forward)
        if not hasattr(decoder, 'original_forward'): setattr(decoder, 'original_forward', decoder.forward)

        self.hooked = True
        
        encoder.forward = VAEHook(encoder, encoder_tile_size, is_decoder=False, **kwargs)
        decoder.forward = VAEHook(decoder, decoder_tile_size, is_decoder=True,  **kwargs)

    def postprocess(self, p:Processing, processed, enabled:bool, *args):
        if not enabled: return

        vae = p.sd_model.first_stage_model
        encoder = vae.encoder
        decoder = vae.decoder
        if isinstance(encoder.forward, VAEHook):
            encoder.forward.net = None
            encoder.forward = encoder.original_forward
        if isinstance(decoder.forward, VAEHook):
            decoder.forward.net = None
            decoder.forward = decoder.original_forward
